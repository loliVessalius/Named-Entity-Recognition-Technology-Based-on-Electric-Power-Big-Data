#!/usr/bin/env python
# coding: utf-8

# ### 1、导包

# In[ ]:


# 导入paddle库
import paddle
import paddle.nn as nn
# 导入paddlenlp的库
from paddlenlp.transformers import  BertModel
from paddlenlp.transformers import BertTokenizer,BertPretrainedModel

from utils import gernate_dic,load_dicts,checkbioes
# 读取数据
from paddlenlp.layers.crf import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss
# import  xlrd2 as xlrd
import re


# ### 2、定义模型初始化方法

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


# ### 3、初始化模型+加载模型

# In[ ]:


id2label,label2id,label_list = load_dicts('data1/tag.dic')
print(id2label)
print(label2id)
print(label_list)


#以下6个参数在在bilstm用到
bigru_hidden_size, num_layers, num_labels, dropout = \
    256,2,len(list(id2label)),0.1

label_num = len(label_list)
no_entity_id = label_num - 1
max_seq_length=128

# 初始化模型
model_name_or_path='bert-base-chinese'
bert = BertModel.from_pretrained(model_name_or_path)
bert_bigru_crf = Bert_BiGRU_crf(bert,label_num,bigru_hidden_size,num_layers,dropout)

# 加载模型
init_checkpoint_path='model/bert_bigru_crf.pdparams'
model_dict = paddle.load(init_checkpoint_path)
bert_bigru_crf.set_dict(model_dict)
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)


# ### 4、解码

# In[ ]:


# 解码
def decode(sentence,preds):
    sentence = sentence.lower()
    word_dict = {}
    end_flag = 0
    tmp_type = ''
    tmp_word = ''
    for index ,item in enumerate(preds):
        if item == 'O' and end_flag == 0:
            continue
        elif item == 'O' and end_flag == 1:
            tmp_word = tmp_word + sentence[index]
        elif item == 'S':
            tmp_type = item.split('-')[1]
            if tmp_type in word_dict:
                word_dict[tmp_type] = word_dict[tmp_type] + ',' + sentence[index]
            else:
                word_dict[tmp_type] = sentence[index]
        elif item.startswith('B'):
            end_flag = 1
            tmp_type = item.split('-')[1]
            tmp_word = sentence[index]
        elif item.startswith('E'):
            end_flag = 0
            tmp_word = tmp_word + sentence[index]
            if len(tmp_word)>0:
                if tmp_type in word_dict:
                    word_dict[tmp_type] = word_dict[tmp_type] + ',' + tmp_word
                else:
                    word_dict[tmp_type] = tmp_word
            tmp_word = ''
            tmp_type = ''
        elif item.startswith('I'):
            if len(tmp_type)==0:
                tmp_type = tmp_type = item.split('-')[1]
                tmp_word = sentence[index]
            else:
                tmp_word = tmp_word + sentence[index]
    
    return word_dict


# ### 5、预测

# In[ ]:


def predict(text, tokenizer=tokenizer, no_entity_id=no_entity_id,max_seq_len=max_seq_length):
    example=list(text)
    example1 = tokenizer(
        example,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)
    bert_bigru_crf.eval()
   
    input_ids=paddle.to_tensor([example1['input_ids']])
    token_type_ids=paddle.to_tensor([example1['token_type_ids']])
    length=example1['seq_len']
    logits = bert_bigru_crf(input_ids, token_type_ids, paddle.to_tensor(length))
    pred = logits.numpy()
    sent = "".join(example)
    tags = [id2label[x] for x in pred[0][1:length-1]]
    str = decode(sent, checkbioes(tags))
    return str


# In[ ]:


result = predict('华东电网公司2007年制定了智能电网发展计划。')
print(result)

result = predict('本次项目完成了信息集成平台建设，发展规划系统建设以及可视化辅助决策系统建设。')
print(result)





