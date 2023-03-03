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
规则：
    if set("一") & set(x):  # 进行与操作
        y = 0
    elif set("J")&set(x):
        y = 1
    elif set("，")&set(x):
        y = 2
    elif set("!")&set(x):
        y = 3
    else:
        y = 4
交叉熵损失函数 第一个参数可能的概率，第二个参数是正确的标签
relu激活函数

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        # __init__()初始化
        super(TorchModel, self).__init__()

        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.classify1 = nn.Linear(vector_dim, 10)     #线性层
        self.classify2 = nn.Linear(10, 5)
        self.activation1 = torch.relu     #sigmoid归一化函数
        self.activation2 = torch.softmax
        self.loss = nn.CrossEntropyLoss()  #loss函数采用均方差损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # x = x.transpose(1,2)                     #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        # x = self.pool(x)                         #  (batch_size, vector_dim, sen_len) ->(batch_size, vector_dim, 1)
        # x = x.squeeze()                          # (batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        x = self.pool(x.transpose(1, 2)).squeeze() #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)
        x = self.classify1(x)                       #(batch_size, vector_dim) -> (batch_size, 1)
        x = self.activation1(x)
        x = self.classify2(x)
        y_pred = self.activation2(x, dim=1)
        label = torch.argmax(y_pred, dim=1)

        # y_pred = self.activation(x)                #(batch_size, 1) -> (batch_size, 1)
        if y is not None:
            y = y.squeeze()
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred,label                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "一二三四五六七八九十JQKA，。、；】【？!@#￥%"  #字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index   #每个字对应一个序号
    vocab['u'] = len(vocab)
    return vocab

# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    #vocab 是词典
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    if set("一") & set(x):  # 进行与操作
        y = 0
    #指定字都未出现，则为负样本
    elif set("J")&set(x):
        y = 1
    elif set("，")&set(x):
        y = 2
    elif set("!")&set(x):
        y = 3
    else:
        y = 4
    # 字典get() 函数返回指定键的值，如果值不在字典中返回默认值。
    # 转换成整数数字
    x = [vocab.get(word, vocab['u']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    count0,count1,count2,count3,count4=0,0,0,0,0
    for i in y:
        if i ==0:
            count0 +=1
        elif i==1:
            count1 += 1
        elif i==2:
            count2 += 1
        elif i==3:
            count3 += 1
        else:
            count4 += 1
    print("本次预测集中共有0类%d个，1类%d个,2类%d个,3类%d个,4类%d个."%(count0, count1,count2,count3,count4))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred,labels = model(x)      #模型预测
        for y_p, y_t, label in zip(y_pred, y, labels):  #与真实标签进行对比
            if label == int(y_t) :
                correct += 1   #负样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 10       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 4         #每个字的维度
    sentence_length = 6   #样本文本长度
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
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
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
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))  # indent:参数根据数据格式缩进显示，读起来更加清晰。
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 4  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    #model.eval()的作用是不启用Batch Normalization 和 Dropout. 相当于告诉网络，目前在eval模式，dropout层不会起作用
    with torch.no_grad():  #不计算梯度
        results,labels = model.forward(torch.LongTensor(x))  #模型预测
    for input_string, result, label in zip(input_strings,results,labels):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, label, float(result[label]))) #打印结果



if __name__ == "__main__":
    main()
    test_strings = ["一二三四五六", "JQKAAJ", "，。、；】【", "uuuuuu"]
    predict("model.pth", "vocab.json", test_strings)
