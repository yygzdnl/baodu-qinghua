#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from operator import itemgetter

"""

实现一个网络完成一个简单nlp任务
判断中文文本中是否有某些特定字符出现

"""
def Bertvocab_built():
    ch_tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-chinese',  # 下载基于 BERT 模型的分词方法的中文字典包
        cache_dir='E:\\AIWork\\badou-qinghua\\38-Dingqingjiang\\20230223_HW2\\ch_vocab',  # 字典的下载位置
        force_download=False  # 不会重复下载
    )  # 执行这条语句时会加载一点时间
    # 获取字典
    vocab = ch_tokenizer.get_vocab()
    return  vocab

# 字符->序号；给语料库贴标签
def dataset_built(txt_path,vocab):
    # 导入语料库
    with open(txt_path, 'r') as f:
        exam_data = f.readlines()  # txt中所有字符串读入data，得到的是一个list
    x_dataset = []
    y_dataset = []
    for exam in exam_data:
        if set("我你您他她它") & set(exam):
            y = 1
        #指定字都未出现，则为负样本
        else:
            y = 0
        exam = [vocab.get(word, vocab['[UNK]']) for word in exam]   #将字转换成序号，为了做embedding
        x_dataset.append(exam)
        y_dataset.append([y])
    return padding(x_dataset), y_dataset

# 补齐为tensor
def padding(x_dataset):
    # 转化为不同维度的tensor
    x_dataset = list(map(lambda x: torch.LongTensor(x), x_dataset))
    # 补齐，True:Batchsize*Tensordim*n, False: T*B*n
    x_dataset = torch.nn.utils.rnn.pad_sequence(x_dataset, batch_first=True)
    return x_dataset

# 将语料库切分成训练集和测试集
def dataset_split(x_dataset, y_dataset, sample_length):
    data_index = [i for i in range(len(x_dataset))]
    random.shuffle(data_index) # random.shuffle没有返回值，直接将data_index乱序
    x_testset = [i for i in itemgetter(*data_index[:sample_length])(x_dataset)] # 200个作为训练集
    y_testset = [i for i in itemgetter(*data_index[:sample_length])(y_dataset)]
    x_trainset = [i for i in itemgetter(*data_index[sample_length:])(x_dataset)] # 剩下的作为训练集
    y_trainset = [i for i in itemgetter(*data_index[sample_length:])(y_dataset)]
    return padding(x_trainset), torch.FloatTensor(y_trainset), padding(x_testset), torch.FloatTensor(y_testset)


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__() # 继承类
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.classify = nn.Linear(vector_dim, 1)     #线性层
        self.activation = torch.sigmoid     #sigmoid归一化函数
        self.loss = nn.functional.mse_loss  #loss函数采用均方差损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        #x = self.pool(x.squeeze())
        x = self.pool(x.transpose(1, 2)).squeeze() #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)
        x = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 1)
        y_pred = self.activation(x)                #(batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, x_testset, y_testset, sample_length):
    model.eval()
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y_testset), sample_length - sum(y_testset)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x_testset)      #模型预测
        for y_p, y_t in zip(y_pred, y_testset):  #与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1   #负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1   #正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    txt_path = 'E:\\AIWork\\badou-qinghua\\38-Dingqingjiang\\20230223_HW2\\exam.txt'
    # 配置参数
    epoch_num = 5        #训练轮数
    batch_size = 20       #每次训练样本个数
    char_dim = 20         #每个字的维度
    sentence_length = 38   #样本文本长度
    learning_rate = 0.1 #学习率
    sample_length = 1000  #测试集样本数
    # 建立字表
    vocab = Bertvocab_built()
    x, y = dataset_built(txt_path, vocab)
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        x_trainset, y_trainset, x_testset, y_testset = dataset_split(x, y, sample_length)
        train_sample = len(x_trainset)  # 每轮训练总共训练的样本总数
        for batch in range(int(train_sample / batch_size)):
            x_batch = x_trainset[batch:(batch+1) * batch_size]
            y_batch = y_trainset[batch:(batch+1) * batch_size]
            optim.zero_grad()    #梯度归零
            loss = model(x_batch, y_batch)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, x_testset, y_testset, sample_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = max([len(i) for i in input_strings])  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['[UNK]']) for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(padding(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i])) #打印结果



if __name__ == "__main__":
    main()
    test_strings = ["他在电影院", "苹果树上长苹果", "看天上有只鸟！", "我不知道。"]
    predict("model.pth", "vocab.json", test_strings)
