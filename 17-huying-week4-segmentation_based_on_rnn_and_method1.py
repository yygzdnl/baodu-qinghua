#coding:utf8

import torch
import torch.nn as nn
import jieba
import numpy as np
import random
import json
from torch.utils.data import DataLoader
import re
import time

"""
基于pytorch的网络编写一个分词模型
我们使用jieba分词的结果作为训练数据
看看是否可以得到一个效果接近的神经网络模型
"""

class TorchModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_rnn_layers, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim) #shape=(vocab_size, dim)
        self.rnn_layer = nn.RNN(input_size=input_dim,
                            hidden_size=hidden_size,   #这是rnn层中的计算
                            batch_first=True,          #表示张量的第一个维度表示是batch_size
                            num_layers=num_rnn_layers,
                            dropout=0.1)               #这里的dropout可加可不加，加入的话，可以丢掉一些前面的记忆
        self.classify = nn.Linear(hidden_size, 2)      #映射到二维
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)  #用交叉熵做损失函数，-100表示不足一个batch_siza的时候补充位置的标记为-100
                                                                 #需要把每一个字的都计算
    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  #input shape: (batch_size, sen_len), output shape:(batch_size, sen_len, input_dim)
        x, _ = self.rnn_layer(x)  #output shape:(batch_size, sen_len, hidden_size)
        y_pred = self.classify(x)   #output shape:(batch_size, sen_len, 2)
        if y is not None:
            # view(-1,2): (batch_size, sen_len, 2) ->  (batch_size * sen_len, 2)
            return self.loss_func(y_pred.view(-1, 2), y.view(-1))   #这句不太懂
                   #view(-1,2)：就是把(batch_size, sen_len, 2)整理为(batch_size * sen_len, 2)形状，
                   #这样在交叉熵看来，就是有batch_size * sen_len个样本，每一个样本是一个二维的，因为一句话中的每个字都要考虑
                   #y.view(-1)是什么意思？？？？  他本来就是一个一维的啊！
        else:
            return y_pred

class Dataset:
    def __init__(self, corpus_path, vocab, max_length):
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.load()

    def load(self):
        self.data = []
        with open(self.corpus_path, encoding="utf8") as f:
            for line in f:
                sequence = sentence_to_sequence(line, self.vocab)
                label = sequence_to_label(line)
                sequence, label = self.padding(sequence, label)
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label)
                self.data.append([sequence, label])
                #使用部分数据做展示，使用全部数据训练时间会相应变长
                if len(self.data) > 10000:
                    break

    #将文本截断或补齐到固定长度
    def padding(self, sequence, label):
        sequence = sequence[:self.max_length]
        sequence += [0] * (self.max_length - len(sequence))
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))    #这里就是把-100作为补充位置的标签
        return sequence, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

#文本转化为数字序列，为embedding做准备
def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab['unk']) for char in sentence]   #查对应的词表，得到词表的词对应的序号
    return sequence

#基于结巴生成分级结果的标注
def sequence_to_label(sentence):
    words = jieba.lcut(sentence)  #用结巴来进行分词，即用结巴的词表，分出这个片段中有哪些词语
    label = [0] * len(sentence)
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer - 1] = 1
    return label

#加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1   #每个字对应一个序号
    vocab['unk'] = len(vocab) + 1
    return vocab

#建立数据集
def build_dataset(corpus_path, vocab, max_length, batch_size):
    dataset = Dataset(corpus_path, vocab, max_length) #diy __len__ __getitem__
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size) #torch
    return data_loader


def main():
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    char_dim = 50         #每个字的维度
    hidden_size = 100     #隐含层维度
    num_rnn_layers = 3    #rnn层数
    max_length = 20       #样本最大长度
    learning_rate = 1e-3  #学习率
    vocab_path = "chars.txt"  #字表文件路径
    corpus_path = "corpus.txt"  #语料文件路径
    vocab = build_vocab(vocab_path)       #建立字表
    data_loader = build_dataset(corpus_path, vocab, max_length, batch_size)  #建立数据集
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)   #建立模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)     #建立优化器
    #训练开始
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y in data_loader:
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    #保存模型
    torch.save(model.state_dict(), "model_week4.pth")
    return

#最终预测
def predict(model_path, vocab_path, input_strings):
    #配置保持和训练时一致
    char_dim = 50  # 每个字的维度
    hidden_size = 100  # 隐含层维度
    num_rnn_layers = 3  # rnn层数
    vocab = build_vocab(vocab_path)       #建立字表
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)   #建立模型
    model.load_state_dict(torch.load(model_path))   #加载训练好的模型权重
    model.eval()
    for input_string in input_strings:
        #逐条预测
        x = sentence_to_sequence(input_string, vocab)
        with torch.no_grad():
            result = model.forward(torch.LongTensor([x]))[0]
            result = torch.argmax(result, dim=-1)  #预测出的01序列
            #在预测为1的地方切分，将切分后文本打印出来
            for index, p in enumerate(result):
                if p == 1:
                    print(input_string[index], end=" ")
                else:
                    print(input_string[index], end="")
            print()

#**********************************************************************
#正向最大匹配的方式进行分词                                                *
#**********************************************************************
#加载词典
def load_word_dict(path):
    max_word_length = 0
    word_dict = {}  #用set也是可以的。用list会很慢
    with open(path, encoding="utf8") as f:
        for line in f:
            word = line.split()[0]
            word_dict[word] = 0
            max_word_length = max(max_word_length, len(word))
    return word_dict, max_word_length

#先确定最大词长度
#从长向短查找是否有匹配的词
#找到后移动窗口
def cut_method1(string, word_dict, max_len):
    words = []
    while string != '':
        lens = min(max_len, len(string))
        word = string[:lens]
        while word not in word_dict:
            if len(word) == 1:
                break
            word = word[:len(word) - 1]
        words.append(word)
        string = string[len(word):]
    return words
#**********************************************************************

if __name__ == "__main__":
     #main()
     input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]
     print("===========用机器学习方式进行分词输出=======")
     predict("model_week4.pth", "chars.txt", input_strings)
     print("===========正向最大匹配方式分词输出=======")
     word_dict, max_len = load_word_dict("dict.txt")
     for i in range(len(input_strings)):
         input_string_1=input_strings[i]
         words = cut_method1(input_string_1, word_dict, max_len)
         print(" ".join(words))

