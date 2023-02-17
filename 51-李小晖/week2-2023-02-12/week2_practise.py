# coding:utf8
import math

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import os

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：
    输入:[x1, x2, x3, x4]
    输出:如果: (2.5 * x1 + 3.6 * x3) > (4.8 * x2 + 1.6 * x4)则为正样本
        否则: 则为负样本
"""
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)  # 线性层1
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)  # 线性层2
        self.linear3 = nn.Linear(hidden_size_2, 1)  # 线性层3
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x1 = self.linear1(x) # (batch_size, input_size) -> (batch_size, hidden_size_1)
        x2 = self.linear2(x1) # (batch_size, hidden_size_1) -> (batch_size, hidden_size_2)
        x3 = self.linear3(x2) # (batch_size, hidden_size_2) -> (batch_size, 1)
        y_pred = self.activation(x3)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 输入:[x1, x2, x3, x4]
# 输出:如果: (2.5 * x1 + 3.6 * x3) > (4.8 * x2 + 1.6 * x4)则为正样本
#     否则: 则为负样本

data_x = []
data_y = []

input_size = 4  # 输入向量维度
hidden_size_1 = 16 # 隐藏层1维度
hidden_size_2 = 8 # 隐藏层2维度

train_sample_num = 20000 # 训练总共的样本总数
eval_sample_num = 2000 # 验证总共的样本总数
test_sample_num = 200 # 测试总共的样本总数
total_sample_num = train_sample_num + eval_sample_num + test_sample_num # 总共的样本数据总数
linear_model_name = "week2-20230212.pth"

isPositive = 0

def build_sample():
    #循环进行，保证正样本和负样本数量差别不大
    global isPositive
    while True:
        x = np.random.random(4) * 5
        if (2.5 * x[0] + 3.6 * x[2]) > (4.8 * x[1] + 1.6 * x[3]):
            if isPositive == 1:
                continue
            else:
                isPositive = 1
                return x, 1
        else:
            if isPositive == -1:
                continue
            else:
                isPositive = -1
                return x, 0
    return None,None#返回非法

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    if total_sample_num <= 0:
        return None,None
    #构建数据集
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    #return X,Y
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# 用来验证每轮模型的准确率
def evaluate(model, eval_x,  eval_y):
    global eval_sample_num # 验证总共的样本总数
    model.eval()
    print("=========本次验证集中共有%d个样本，%d个正样本，%d个负样本=========" % (len(eval_x), sum(eval_y), eval_sample_num - sum(eval_y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(eval_x)  # 模型预测
        #sigmod函数返回值(0, 1)
        for y_p, y_t in zip(y_pred, eval_y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("=========正确预测个数：%d, 正确率：%f=========" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def train(train_x, train_y, eval_x,  eval_y):
    # 配置参数
    global input_size # 输入向量维度
    global hidden_size_1 # 隐藏层1维度
    global hidden_size_2 # 隐藏层2维度

    epoch_num = 100  # 训练轮数
    batch_size = 200  # 每次训练样本个数
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size, hidden_size_1, hidden_size_2)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    #样本总数
    train_count = len(train_x)
    print("=========训练样本总数:%d=========" % (train_count))
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_count // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========第%d轮平均loss:%f=========" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, eval_x,  eval_y)  # 验证本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
        #判断是否需要提前结束
        if float(np.mean(watch_loss)) <= 0.001:
            break
    # 保存模型
    torch.save(model.state_dict(), linear_model_name)
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, test_x, test_y):
    global input_size # 输入向量维度
    global hidden_size_1 # 隐藏层1维度
    global hidden_size_2 # 隐藏层2维度

    current_index = 0 # 当前序列号
    pred_right_count = 0 # 预测正确的数量
    #样本总数
    predict_count = len(test_x)
    print("=========测试样本总数:%d=========" % (predict_count))

    model = TorchModel(input_size, hidden_size_1, hidden_size_2)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())  #打印参数信息

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(test_x))  # 模型预测
    for vec, res in zip(test_x, result):
        #每一次预测的准确值和概率值
        if round(float(res)) == test_y[current_index]:
            pred_right_count += 1

        current_index += 1
        print("=========测试输入：%s, 预测类别：%d, 概率值：%f=========" % (vec, round(float(res)), res))  # 打印结果
    # 对测试样本总的准确率输出
    print("*********测试整体正确率: ( %d / %d = %f%%)*********" % (pred_right_count, predict_count, pred_right_count / predict_count))  # 打印结果

if __name__ == "__main__":
    #train_sample_num = 20000 # 训练总共的样本总数
    #eval_sample_num = 2000 # 验证总共的样本总数
    #test_sample_num = 2000 # 测试总共的样本总数
    #total_sample_num# 总共的样本数据总数

    #使用同一种规律产生训练、验证和测试数据
    data_x, data_y = build_dataset(total_sample_num)

    train_x = data_x[:train_sample_num]
    train_y = data_y[:train_sample_num]

    eval_x = data_x[train_sample_num:train_sample_num + eval_sample_num]
    eval_y = data_y[train_sample_num:train_sample_num + eval_sample_num]

    test_x = data_x[-eval_sample_num:]
    test_y = data_y[-eval_sample_num:]

    print("*********************正在执行主流程，请稍后*********************")
    #模型文件是否存在
    print("*********************模型准备中，请稍后*********************")
    is_need_train = True
    if os.path.exists(linear_model_name):
        ret = input("@ 模型文件已经存在，是否需要重新训练?(输入\"1\"表示需要训练，\"0\"表示不需要训练)？")
        if int(ret) == 0:# 不需要训练
            is_need_train = False
    if is_need_train == True:
        print("*********************模型训练中，请稍后*********************")
        train(train_x, train_y, eval_x, eval_y)
    #进行预测
    print("*********************模型预测中，请稍后*********************")
    predict(linear_model_name, test_x, test_y)
    print("*********************主流程执行完毕，谢谢使用*********************")
