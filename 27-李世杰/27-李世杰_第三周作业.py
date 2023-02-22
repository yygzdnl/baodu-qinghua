# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本
第二周作业：
自己设计新的正负样本规律，更改模型结构，完成模型训练。
设计为四分类：
    修改：
        x 为五维向量，
          if x[0] > x[1] and x[1] > x[2]:
            return x, 0
        elif x[0]< x[1] and x[1] <x[2]:
            return x, 1
        elif x[0] > x[1] and x[0] <x[2]:
            return x, 2
        elif x[0] < x[1] and x[1] > x[2]:
            return x, 3
        else:
            return build_sample() 如果存在上述其他情况，重新生成数据
        增加了一层线性层和激活函数
        
        保存准确率最高的模型。best_model.pth
    
    激活函数采用 softmax、relu
    relu作为中间层的激活函数，在实验效果上相对于sigmoid效果更好，
    Relu函数的导数计算更快，所以使用梯度下降时比Sigmod收敛起来要快很多。
    
    Sigmoid优点：具有很好的解释性，将线性函数的组合输出为0，1之间的概率。
    Sigmoid缺点：（1）激活函数计算量大，反向传播求梯度时，求导涉及除法。
                （2）反向传播时，在饱和区两边导数容易为0，即容易出现梯度消失的情况，从而无法完成深层网络的训练。
    
    损失函数采用 交叉熵损失函数
    
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        # 继承TorchModel的基类。就是nn.Module.__init__()
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 20)  # 线性层
        self.linear2 = nn.Linear(20, 4)
        self.activation1 = torch.relu  # sigmoid归一化函数
        self.activation2 = torch.softmax
        # self.loss = nn.functional.mse_loss  # loss函数采用均方差损失
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x1 = self.linear1(x)  # (batch_size, input_size) -> (batch_size, 1)
        x1_a = self.activation1(x1)  # (batch_size, 1) -> (batch_size, 1)
        x2 = self.linear2(x1_a)
        output = self.activation2(x2, dim=1)
        label = torch.argmax(output, dim=1)
        if y is not None:
            y = y.reshape([y.size()[0], ])
            return self.loss(output, y)  # 预测值和真实值计算损失
        else:
            return output,label  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)
    if x[0] > x[1] and x[1] > x[2]:
        return x, 0
    elif x[0]< x[1] and x[1] <x[2]:
        return x, 1
    elif x[0] > x[1] and x[0] <x[2]:
        return x, 2
    elif x[0] < x[1] and x[1] > x[2]:
        return x, 3
    else:
        return build_sample()


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])

    X = np.array(X)
    Y = np.array(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    num0, num1, num2, num3 = 0, 0, 0, 0
    for i in y:
        if i == 0:
            num0 += 1
        elif i == 1:
            num1 += 1
        elif i == 2:
            num2 += 1
        else:
            num3 += 1
    print("本次预测集中共有%d个0样本，%d个1样本，%d个2样本，%d个3样本" % (num0, num1, num2, num3))
    correct, wrong = 0, 0
    with torch.no_grad():
        # 有真实标签  返回loss值，无真实标签，返回预测值
        y_pred,lable = model(x)  # 模型预测
        for y_p, y_t in zip(lable, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1  # 负样本判断正确
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 15  # 训练轮数
    batch_size = 15  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    accs=[]
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果

        log.append([acc, float(np.mean(watch_loss))])
        if len(accs)!=0 and acc > max(accs):
            best_model = model.state_dict()
        accs.append(acc)
    # 保存模型
    torch.save(best_model, "best_model.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result,labels = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res, label in zip(input_vec, result,labels):
        # round()函数默认小数位数为0，表示该函数将返回最接近的整数，round(number,digits)
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, label,res[label]) ) # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.47889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.99871392],
                [0.1349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    predict("model.pth", test_vec)
