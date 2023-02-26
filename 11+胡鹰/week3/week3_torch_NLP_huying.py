#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现
********************************************************************************
*如果某个长度为10的字符串中出现字符b、d、5、n，则判定该字符串样本为1，否则判定该字符串样本为0
********************************************************************************

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
             #vector_dim表示一个字符需要转化为1个多少维的向量
             #embedding层的作用：将字符串转化为对应的矩阵，矩阵行数为字符串的长度len(vocab)，矩阵列数为我们自己设定的vector_dim维度
             #每个字符用该矩阵中的一个行的向量来表示。这样避免了标量的相加，而且空间向量几乎不会重合！
        self.pool = nn.AvgPool1d(sentence_length)   #池化层
             #AvgPool1d层：将embedding层转化来的矩阵，进行压缩池化。比如：[[x,x,x,x,x],
             #                                                     [v,v,v,v,v],
             #                                                     [c,c,c,c,c]]  3*5的矩阵，代表了3个字符的字符串
             #                                                                   每个字符用一个1*5的向量表示
             #按竖向x,v,c按个数求平均，得到一个1*5的向量。这个向量就代表了这个字符串池化后的，它再经过一个5*1线性层w,就可以得到一个标量
        self.classify = nn.Linear(vector_dim, 1)     #线性层
             #就是通过一个5*1的向量来把前面池化后的1*5向量进行转换为一个标量
        self.activation = torch.sigmoid     #sigmoid归一化函数
             #就会将前面得到的标量进行归一化，使其得到0-1之间的一个数字，进而为后面识别出是0还是1做准备
        self.loss = nn.functional.mse_loss  #loss函数采用均方差损失
             #用均方差来计算测归一化后的数字与真实值的差值，确定损失为多少
        #self.loss = nn.CrossEntropyLoss()

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
                   #得到20个为分段，每个字符串长度为10，每个字符40维的矩阵
        x = self.pool(x.transpose(1, 2)).squeeze() #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)
                   #池化层进行降维。因为池化层只对最后一个降维，这里的最后一维是每个字符对应的40维的一维向量，所以需要做矩阵转置
                   #因为我们的目标是对每一个样本做池化，这个样本中有10个字符，要把这10个字符降为1个字符，但还是40维；
                   #转置以后，把40放在中间，最后1位是10，所以才能把10个字符降为1个字符来方便后续计算
                   #x.transpose(1, 2)表示转置矩阵的第2位和第3位，因为0代表第1位（对应的整体样本的分段长度，它不需要降）
                   #举例：整个样本的每个分段长度为3个样本，每个样本为2个字符的字符串，每个字符embedding为1个3维的向量，那么得到：
        x = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 1)
                   #通过一个类似于线性层的，乘以vector_dim*1的矩阵，得到一个标量
        y_pred = self.activation(x)                #(batch_size, 1) -> (batch_size, 1)
                   #激活函数，使得可以非线性化
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

#******************************************************************************************
#下面是生成一个字典：每个字符对应1个下标，用这个下标去取前面embeding层给我们生成的矩阵中对应字符的那一行向量
#******************************************************************************************
#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"  #字符集
    vocab = {}                            #字典
    for index, char in enumerate(chars):
        vocab[char] = index               #每个字对应一个序号
    vocab['unk'] = len(vocab)             #不认识的字符就定义为unk对应的序号
    return vocab

#***********************************************************************************
#1个样本的生成函数：穿入字典、需要生成的样本长度，自动随机产生字符串，并判断该字符串是1还是0，一并返回
#***********************************************************************************
#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    if set("bd5n") & set(x):         #这行代码是：abd其中任何一个和x字符串做与操作，为真，则认为样本为1
        y = 1
    #elif set("xyz") & set(x):
    #    y = 2
    #指定字都未出现，则为负样本
    else:
        y = 0
    # 将字转换成序号，为了做embedding
    x = [vocab.get(word, vocab['unk']) for word in x]    #这句代码不会很懂
                                                         #vocab.get()方法，应该是拿到字典中keys对应的值
    return x, y   #把样本和判断值组合后一起返回

#*************************************************************************
#创建训练需要的样本，输入需要的样本个数、来源字典、每个字符串样本的长度，返回torch的张量
#*************************************************************************
#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)        #build_sample函数就是生成1个字符串和判断值
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#********************************************************************************
#实例化1设定的模型
#vocab字典，char_dim表示1个字符需要转换为一个多少维的向量，sentence_length表示1个字符串的长度
#********************************************************************************
#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#*************************
#用来测试每1轮模型学习后的准确率
#*************************
def evaluate(model, vocab, sample_length):
    model.eval()                                      #这行代码就代表是在测试了，
                                                      # torch会自动停止model中的embedding层，以防止训练和测试时的字符对应向量不同
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():      #测试的时候不会再更新梯度，就代表不是在学习。是否在进行学习，关键就是看梯度会否会被更新
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比，用0.5来作为正确和错误的分界
            if float(y_p) < 0.5 and int(y_t) == 0:     #预测值小于0.5，真实值为0，说明预测值是偏向真实值的，判断为正确的负样本
                correct += 1   #负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:  #预测值大于0.5，真实值为1，说明预测值是偏向真实值的，判断为正确的正样本
                correct += 1   #正样本判断正确
            #elif float(y_p) >= 0.5 and float(y_p) < 1.5 and int(y_t) == 1:
            #     correct += 1
            #elif float(y_p) >=1.5 and int(y_t) == 2:
            #     correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


#****************************************************
#学习过程
#****************************************************
def main():
    #配置参数
    epoch_num = 30        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 800    #每轮训练总共训练的样本总数
    char_dim = 40         #每个字的维度
    sentence_length = 10   #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss     ？？？自动调用forword方法，返回loss值
            loss.backward()      #计算梯度      ？？？自动利用loss值调用back方法计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss))) #np.mean（）取平均值
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型训练好以后的权重
    torch.save(model.state_dict(), "model_nlp_week3.pth")
    # 保存词表---词表必须保存，这样才能和训练得到的结果对应起来
    writer = open("vocab_week3.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))  #json.dumps（）这个函数是写入？？
    writer.close()
    return

#*************************************************
#使用训练好的模型做预测
#*************************************************
def predict(model_path, vocab_path, input_strings):
    char_dim = 40  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
                                                          #[vocab[char] for char in input_string]就是：
                                                          #在字典中去找char并返回keys对应的值
    model.eval()           #开启测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i])) #打印结果
                                                               #result[i]这里不是很懂？



if __name__ == "__main__":
    main()
    test_strings = ["ffvfeeggn0", "wwssfggggg", "rqwdbgnyjk", "jakwwwy9u7"]
    predict("model_nlp_week3.pth", "vocab_week3.json", test_strings)
