

import torch
import torch.nn as nn
import numpy as np
import random
import json
from transformers import BertModel, BertTokenizer


hidden_size = 768
num_heads = 12
num_layer = 12
vocab_size = 21128
max_len = 512


class BertEmbedding(nn.Module):
    def __init__(self, hidden_size, max_len, vocab_size):
        super(BertEmbedding, self).__init__()
        # embedding 层
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.token_embedding = nn.Embedding(2, hidden_size)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        # embedding layer norm
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size)

    def forward(self, x):
        # x:(batch_size, seq_len)
        # word_embedding: (batch_size, seq_len, hidden_size)
        word_embedding = self.word_embedding(x)

        toke_type = torch.zeros((32, 512))  # 暂定全是一个句子
        # token_embedding:(batch_size, seq_len, hidden_size)
        token_embedding = self.token_embedding(toke_type)

        position = torch.arange(x.shape[1]).expand(x.shape[0], -1)
        # position_embedding:(batch_size, seq_len, hidden_size)
        position_embedding = self.position_embedding(position)

        # embedding:(batch_size, seq_len, hidden_size)
        embedding = word_embedding + token_embedding + position_embedding
        embedding = self.layer_norm(embedding)

        return embedding


class BertEncoder(nn.Module):
    def __init__(self, num_heads, hidden_size):
        super(BertEncoder, self).__init__()
        # transformer (encoder 层)
        # attention层
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.3)
        self.attention_linear = nn.Linear(hidden_size, hidden_size)
        self.attention_layer_norm = nn.LayerNorm(normalized_shape=hidden_size)
        # 前向反馈层
        self.feed_forward_linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.feed_forward_linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.feed_forward_layer_norm = nn.LayerNorm(normalized_shape=hidden_size)

    def forward(self, x):
        # x:(batch_size, seq_len, hidden_size)
        # attention:(batch_size, seq_len, hidden_size)
        attention = self.attention(x, x, x)
        # attention_linear:(batch_size, seq_len, hidden_size)
        attention_linear = self.attention_linear(attention)
        # attention_norm:(batch_size, seq_len, hidden_size)
        attention_norm = self.attention_layer_norm(attention + attention_linear)

        # feed_out:(batch_size, seq_len, hidden_size)
        feed_out = self.feed_forward_linear2(nn.GELU(self.feed_forward_linear1(attention_norm)))
        # feed_norm:(batch_size, seq_len, hidden_size)
        feed_norm = self.feed_forward_layer_norm(feed_out + attention_norm)

        return feed_norm


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()
        # 输出层，做池化处理
        self.pooler = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x : (seq_len, seq_len, hidden_size)
        # x = [:, 0] # 取[CLS]的操作
        # out : (batch_size,  hidden_size)
        out = nn.Tanh(self.pooler(x))
        return out


class BertModelTorch(nn.Module):
    def __init__(self, num_heads, hidden_size, max_len, vocab_size):
        super(BertModelTorch, self).__init__()

        # embedding 层
        self.embedding = BertEmbedding(hidden_size, max_len, vocab_size)
        # transformer (encoder 层)
        self.encoder = BertEncoder(num_heads, hidden_size)
        # output 层
        self.output = BertPooler(hidden_size)

    def forward(self, x):
        # x :[batch_size, seq_len]
        # embdded_x.shape :[batch_size, seq_len, hidden_size]
        embdded_x = self.embedding(x)

        sentence_out = self.encoder(embdded_x)

        pool_out = self.output(sentence_out[0])

        return sentence_out, pool_out


if __name__ == "__main__":
    bert = BertModelTorch(num_heads, hidden_size, max_len, vocab_size)
    
    # bert_base 模型参数量计算
    # emdedding 层：
    #    词向量(21128 * 768 ) + segment (2 * 768) + position (512 * 768)  = 16,621,056
    #    layer_norm：w(768) + b (768) = 1536
    #    合计 ： 16,621,056 + 1536 = 16,622,592 
    #
    # encoder 层：
    #    attention :
    #       Q ：w (768*768) + b(768) =590,592
    #       K:：w (768*768) + b(768)= 590,592
    #       V：w (768*768) + b(768)= 590,592
    #       linear： w (768*768) + b（768）= 590,592
    #       layer_norm： w (768) + b(768)=1536
    #       合计：590,592 * 4 + 1536 = 2,363,904
    #
    #    feed forward：
    #      linear1： w(768    *  3072) + b(768) = 2,360,064
    #      linear2： w(3072  *  768) + b(768*4=3072) = 2,362,368
    #      layer_norm:：w (768) + b(768)=1536
    #      合计：2,362,368 +  2,360,064 + 1536 = 4,723,968
    
    #    单层合计： 2,363,904 + 4,723,968 = 7,087,872
    #    所有层合计：12 *  7,087,872 = 85,054,464
    
    # 输出层：
    #    pooler：w(768*768) + b(768) = 590,592
    #
    # 总的参数量：16,622,592 + 85,054,464 + 590,592 = 102,267,648 ≈ 100M
