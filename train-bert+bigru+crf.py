#!/usr/bin/env python
# coding: utf-8

# ### 1、导包

# In[ ]:


import os
import time
from functools import partial
# 导入paddle库
import paddle
import paddle.nn as nn
from paddlenlp.transformers import  BertModel
from paddle.io import DataLoader
# 导入paddlenlp的库
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.transformers import BertTokenizer,BertPretrainedModel
from paddlenlp.data import Stack, Tuple, Pad, Dict

# 读取数据
from paddlenlp.datasets import load_dataset
from paddlenlp.layers.crf import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss

# 导入自定义的工具类
from utils import * 


# In[ ]:


# 从dev.txt和train.txt文件生成dic
gernate_dic('data1/dev.txt', 'data1/train.txt', 'data1/tag.dic')
id2label,label2id,label_list = load_dicts('data1/tag.dic')
print(id2label)
print(label2id)
print(label_list)


# ### 2、把文本和label映射成id，处理成模型的输入的形式

# In[ ]:


label_map=label2id
def read(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        # 跳过列名
        for line in f.readlines():
            words, labels = line.strip('\n').split('\t')
            words = words.split()
            labels = labels.split()
            labels=[label_map[item] for item in labels]
            yield {'tokens': words, 'labels': labels}

# data_path为read()方法的参数
train_ds = load_dataset(read, data_path='data1/train.txt',lazy=False)
dev_ds = load_dataset(read, data_path='data1/dev.txt',lazy=True)


# In[ ]:


def convert_example_to_feature(example, tokenizer, no_entity_id,
                              max_seq_len=512):
    labels = example['labels']
    example = example['tokens']
    tokenized_input = tokenizer(
        example,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    # -2 for [CLS] and [SEP]
    if len(tokenized_input['input_ids']) - 2 < len(labels):
        labels = labels[:len(tokenized_input['input_ids']) - 2]
    tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
    tokenized_input['labels'] += [no_entity_id] * (
        len(tokenized_input['input_ids']) - len(tokenized_input['labels']))
    return tokenized_input


# 利用DataLoader将把处理好的数据输送给模型，下面是训练集和测试集的Dataloader构建过程。

# ### 3、加载数据集train_data_loader和dev_data_loader

# In[ ]:


max_seq_length=128
batch_size=16
label_num = len(label_list)
no_entity_id = label_num - 1

#以下4个参数在在bilstm用到
gru_hidden_size, num_layers, num_labels, dropout = 256,2,len(list(id2label)),0.1

model_name_or_path='bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

trans_func = partial(
        convert_example_to_feature,
        tokenizer=tokenizer,
        no_entity_id=no_entity_id,
        max_seq_len=max_seq_length)
        
train_ds = train_ds.map(trans_func)
dev_ds = dev_ds.map(trans_func)

ignore_label = label_num - 1

batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int32'),  # input
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int32'),  # segment
        'seq_len': Stack(dtype='int64'),  # seq_len
        'labels': Pad(axis=0, pad_val=ignore_label, dtype='int64')  # label
    }): fn(samples)


train_batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

train_data_loader = DataLoader(dataset=train_ds,collate_fn=batchify_fn,num_workers=0,batch_sampler=train_batch_sampler,
        return_list=True)
dev_data_loader = DataLoader(dataset=dev_ds,collate_fn=batchify_fn,num_workers=0,batch_size=batch_size,return_list=True)


# ### 4、定义模型初始化方法

# In[ ]:


class Bert_BiGRU_crf(BertPretrainedModel):

    def __init__(self, bert,num_classes, gru_hidden_size, num_layers,dropout):
        super(Bert_BiGRU_crf, self).__init__()
        self.num_classes = num_classes
        # 初始化bert
        self.bert = bert  
        # 初始化双向的lstm
        self.gru = nn.GRU(self.bert.config["hidden_size"],gru_hidden_size,num_layers,direction='bidirect',dropout=dropout)
        
        self.fc = nn.Linear(gru_hidden_size * 2, self.num_classes)

        # crf层
        self.crf = LinearChainCrf(
            self.num_classes, crf_lr=100, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(
            self.crf.transitions, with_start_stop_tag=False)

    def forward(self,
                input_ids,
                token_type_ids,
                lengths=None,
                labels=None):
        sequence_out, _ = self.bert(input_ids,
                                    token_type_ids=token_type_ids)
        lstm_output, _ = self.gru(sequence_out)
        emission = self.fc(lstm_output)
  
        if labels is not None:
            loss = self.crf_loss(emission, lengths, labels)
            return loss
        else:
            _, prediction = self.viterbi_decoder(emission, lengths)
            return prediction


# ### 5、配置全局参数

# In[ ]:


# 设置epoch
num_train_epochs=12
warmup_steps=0

max_steps=-1
# 优化器的超参数
learning_rate=5e-5
adam_epsilon=1e-8
weight_decay=0.0

global_step = 0
# 日志输出的step数
logging_steps=30

tic_train = time.time()
save_steps=20
output_dir='model'
os.makedirs(output_dir,exist_ok=True)

# Define the model netword and its loss
last_step = num_train_epochs * len(train_data_loader)

num_training_steps = max_steps if max_steps > 0 else len(train_data_loader) * num_train_epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps,
                                         warmup_steps)


# ### 6、初始化模型

# In[ ]:


bert = BertModel.from_pretrained("bert-base-chinese")
bert_bigru_crf = Bert_BiGRU_crf(bert,label_num,gru_hidden_size,num_layers,dropout)

decay_params = [
        p.name for n, p in bert_bigru_crf.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
# 设置优化器
optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=adam_epsilon,
        parameters=bert_bigru_crf.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)
# 设置损失函数
loss_fct = nn.loss.CrossEntropyLoss(ignore_index=ignore_label)
# 设置评估函数
metric = ChunkEvaluator(label_list=label_list,suffix=True)


# ### 7、模型评估方法

# In[ ]:


@paddle.no_grad()
def evaluate_v1(model, metric, data_loader):
    model.eval()
    metric.reset()
    for input_ids, seg_ids, lens, labels in data_loader:
        preds = model(input_ids, seg_ids, lengths=lens)
        n_infer, n_label, n_correct = metric.compute(lens, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("[EVAL] Precision: %f - Recall: %f - F1: %f" % (precision, recall, f1_score))
    model.train()
    return precision


# ### 8、模型训练

# In[ ]:


global_step=0
best_pre = 0
for epoch in range(num_train_epochs):
    for step, batch in enumerate(train_data_loader):
        global_step += 1
        input_ids, token_type_ids, lengths, labels = batch
        loss = bert_bigru_crf(input_ids, token_type_ids, lengths=lengths, labels=labels)
        avg_loss = paddle.mean(loss)
        if global_step % logging_steps == 0:
            print("global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, avg_loss,
                       logging_steps / (time.time() - tic_train)))
            tic_train = time.time()
        avg_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

    pre = evaluate_v1(bert_bigru_crf, metric, dev_data_loader)
    # 如果pre比历史记录高，则保存新的模型
    if pre>best_pre:
        model_path=os.path.join(output_dir,"bert_bigru_crf.pdparams")
        paddle.save(bert_bigru_crf.state_dict(),model_path)
        tokenizer.save_pretrained(output_dir)
        best_pre = pre

