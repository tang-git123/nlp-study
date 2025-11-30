#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def __distance(p1, p2):
    #计算两点间距
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)
    return pow(tmp, 0.5)


def calculate_distance(center, vectors):
    tmp_dis = 0
    for vector in vectors:
        # 计算每个句子向量到中心点的距离，并相加
        tmp_dis += __distance(vector, center)

    # 返回平均距离
    return tmp_dis / len(vectors)


def main():
    model = load_word2vec_model(r"model.w2v") # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)   # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):  # 取出句子，向量和标签
        # kmeans.cluester_centers_ #每个聚类中心
        sentence_label_dict[label].append(sentence)         # 同标签的句子放到一起
        vector_label_dict[label].append(vector)             # 同标签的向量放到一起

    distance_label = []
    for label in range(n_clusters):
        distance = calculate_distance(kmeans.cluster_centers_[label], vector_label_dict[label])  # 计算每个label 的平均距离
        distance_label.append([label, distance])  # 保存每个label的距离

    new_distance_label = sorted(distance_label, key=lambda x: x[1])  # 对label的距离进行从小到大排序
    # 根据排序后的label打印
    for item in new_distance_label:
        print("cluster %s , label_distance %.6f:" % (item[0], item[1]))
        sentences = sentence_label_dict[item[0]]
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")


if __name__ == "__main__":
    main()

