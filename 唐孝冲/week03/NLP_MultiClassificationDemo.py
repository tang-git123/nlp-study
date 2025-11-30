import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import random

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
根据文本中特定字符出现位置下标进行文本的多分类

"""


# RNN模型定义
class TorchRNN(nn.Module):
    def __init__(self, vocab_size, vector_dim, hidden_size, output_size):
        super(TorchRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True)
        self.classify = nn.Linear(hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)
        output = x[:, -1, :]  # (batch_size, hidden_size)
        y_pred = self.classify(output)  # (batch_size, output_size)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=1)  # 输出预测结果


# 构建词表
def build_vocab():
    chars = "中国你我他@abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0, "unk": 1}
    for index, char in enumerate(chars):
        vocab[char] = index + 2  # 每个字对应一个序号

    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    chars = "你我他abcdefghijklmnopqrstuvwxyz"
    # 随机选择sentence_length -1 个不重复的字符
    str_list = random.sample(chars, sentence_length - 1)

    # 把 @ 字符插入字符串的随机位置
    idx = random.randint(0, len(str_list))
    str_list.insert(idx, "@")
    # 将字符按照此表转换为数字，方便embedding
    str_num = [vocab.get(char, vocab['unk']) for char in str_list]
    return str_num, idx


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, hidden_size, output_size):
    model = TorchRNN(len(vocab), char_dim, hidden_size, output_size)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 30  # 每个字的维度
    hidden_size = 64  # 隐藏层的维度
    sentence_length = 10  # 文本长度
    learning_rate = 0.005  # 学习率
    output_size = 10  # 10个类别
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, hidden_size, output_size)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optimizer.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model_multi.pth")
    # 保存词表
    writer = open("vocab_multi.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 30  # 每个字的维度
    hidden_size = 64  # 样本文本长度
    output_size = 10
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, hidden_size, output_size)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  # 将输入序列化

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        # print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i]))  # 打印结果
        print("输入：%s, 预测类别：%s " % (input_string, torch.argmax(result[i])))  # 打印结果


if __name__ == "__main__":
    test_chars = ["ze@uhxjdry",
                  "@我be他shjlr",
                  "o@n你rpg他y我",
                  "rnva@pm他qt",
                  "yuvncwlsf@",
                  "p他skmjo@we",
                  "fqpjdi@yak",
                  "e他w@ndxsjo",
                  "jhyqgmnb@c",
                  "dpcar@buml"]

    # main()
    predict("model_multi.pth", "vocab_multi.json", test_chars)

