import torch
import numpy as np
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import json

'''
判断一个字符串中是否有指定字符
有“abc中国”则为正样本，否则为负样本

"abcdle"     -> 映射成字符长度*维度的随机向量 -> n*6*5
按维度求平均 -> 映射成                        -> n*1*5
线性层       -> w*x+b                         -> n*1*1
激活         -> sigmoid归一化函数             -> n*1*1

'''

class NlpModel(nn.Module):
    def __init__(self, text_dim, sentence_length, vocab):
        super(NlpModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), text_dim)
        self.pool = nn.AvgPool1d(sentence_length)
        self.layer = nn.Linear(text_dim,1)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        x = self.embedding(x)       #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = x.transpose(1,2)        #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        x = self.pool(x)            #(batch_size, vector_dim, sen_len) -> (batch_size, vector_dim, 1)
        x = x.squeeze()             #(batch_size, vector_dim, 1)  -> (batch_size, vector_dim)
        x = self.layer(x)           #(batch_size, vector_dim) -> (batch_size, 1)
        y_pred = self.activation(x) #(batch_size, 1) -> (batch_size, 1)

        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

#读取词表
def read_vab():
    return json.load(open("vab.json", "r", encoding="utf8"))  # 加载字符表

#随机构建样本
def build_sample(vocab,sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if set("abc中国") & set(x):
        y = 1
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y

#构建样本集
def build_dataset(batch_size, vocab, sentence_length):
    x_data = []
    y_data = []
    for i in range(batch_size):
        x,y = build_sample(vocab, sentence_length)
        x_data.append(x)
        y_data.append([y])
    return torch.LongTensor(np.array(x_data)),torch.FloatTensor(np.array(y_data))

# 测试模型
def evaluate(model, vocab, sentence_length):
    model.eval()            #进入测试模式
    test_sample_num = 500
    x,y = build_dataset(test_sample_num, vocab, sentence_length)    #测试样本

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)       #预测值
        for y_p, y_true in zip(y_pred, y):
            if float(y_p) > 0.5:
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 准确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def train():
    epoch_num = 15
    batch_size = 200
    train_sample = 3000     # 训练样本数
    text_dim = 6            # 输入向量维度
    sentence_length = 7     #文本长度
    learning_rate = 0.01    # 学习率

    vocab = read_vab()
    model = NlpModel(text_dim, sentence_length, vocab)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []
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

    # 保存模型参数
    torch.save(model.state_dict(), "model.pth")

    # 画图
    # print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="accuracy")  # 画accuracy曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 6  # 每个字的维度
    sentence_length = 7  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = NlpModel(char_dim, sentence_length, vocab)        #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化

    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i])) #打印结果

if __name__ == "__main__":
    train()

    test_strings = ["abcfwww", "abcdfg中", "rqwdhg天", "gkwwn中国"]
    predict("model.pth", "vab.json", test_strings)


