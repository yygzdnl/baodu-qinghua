
#设计新的 正负样本 规律，更改模型结构，完成训练模式

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pythorch框架编写模型训练机器
规律：X是一个4维向量，如果前两个数之和>后两数之后，则为正样本，反之为负样本
"""


class TorchModel(nn.Module):
    def __init__(self,input_size):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size,1)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    def forward(self,x,y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(train_sample):
    X = []
    Y = []
    for i in range(train_sample):
        x,y = build_num()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X),torch.FloatTensor(Y)

#随机生成一个四维向量，if a+b > c+d  认为是正样本1，反之0
def build_num():
    x=np.random.random(4)
    if x[0] + x[1] > x[2] + x[3] :
        return x, 1
    else:
        return x, 0


def evalate(model):
    model.eval()
    x, y = build_dataset(100)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), 100 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 4
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


#基础配置
def main ():
    epoch_num = 100
    batch_size = 20
    train_sample = 2000  #每轮训练总共训练的样本总数
    input_size =  4
    learning_rate = 0.001

    #建立模型
    model = TorchModel(input_size)

    #选择优化器
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log = []  # 记录用
    #创建训练集
    train_x,train_y = build_dataset(train_sample)

    #训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(train_sample // batch_size):     #100
            x = train_x[i * batch_size: (i + 1) * batch_size]
            y = train_y[i * batch_size: (i + 1) * batch_size]
            optim.zero_grad()  # 梯度归零
            loss = model(x,y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("======\n测试第%d轮平均loss:%f" %(epoch + 1,np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.hxp")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

if __name__ == "__main__":
    main()
    test_vec = [[0.47889086,0.15229675,0.31082123,0.03504317],
                [0.94963533,0.5524256,0.95758807,0.95520434],
                [0.78797868,0.67482528,0.13625847,0.34675372]]
    predict("model.hxp", test_vec)


