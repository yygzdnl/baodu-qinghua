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
规律：x是一个10维向量，如果 (x[0]+x[5])*abs(x[1]-x[6])/2 > (x[9]+x[4])*abs(x[8]-x[3])，则为正样本，否则为负样本
经测试，利用np.around(np.random.random(10), 3) * 1000生成的随机样本中，符合该规律的正负样本比例接近1:1，适合做训练
调用test_dataset_CW_Ratio()方法可以查看正负样本比例
np.around(np.random.random(10), 3) * 1000 生成一个10维向量，每一维取值为[0,999)的整数 把小数转换为整数便于计算

"""

torch.manual_seed(0)  # 设置随机种子，确保每次生成的随机数相同


class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
        super(TorchModel, self).__init__()
        # 设置4个线性层
        self.linear1 = nn.Linear(input_size, hidden_size1)  # 线性层
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)  # 线性层
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)  # 线性层
        self.linear4 = nn.Linear(hidden_size3, 1)

        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # 网络结构的公式为：y_pred = w1 * x ** 3 + w2 * x ** 2 + w3 * x + w4
        x = self.linear1(x * x * x)  # (batch_size, input_size) -> (batch_size, hidden_size1)
        x = self.activation(x)
        x = self.linear2(x * x)  # (batch_size, hidden_size1) -> (batch_size, hidden_size2)
        x = self.activation(x)
        x = self.linear3(x)  # (batch_size, hidden_size2) -> (batch_size, hidden_size3)
        x = self.activation(x)
        x = self.linear4(x)  # (batch_size, hidden_size3) -> (batch_size, 1)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个10维向量，如果(x[0]+x[5])*abs(x[1]-x[6])/2 > (x[9]+x[4])*abs(x[8]-x[3])，则为正样本，否则为负样本
def build_sample():
    x = np.around(np.random.random(10), 3) * 1000
    if (x[0] + x[5]) * abs(x[1] - x[6]) / 2 > \
            x[9] + x[4] * abs(x[8] - x[3]):
        return x, 1
    else:
        return x, 0


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# test 用于测试生成的数据集中正负样本的比例是否接近1:1，epoch是测试的轮次，
# 最后将epoch轮生成的数据集中正负样本的比例以列表的形式打印出来 以验证正负样本比是否接近1:1
def test_dataset_CW_Ratio(total_sample_num, epoch):
    X = []  # 存放生成的x向量
    Yc = []  # 存放正样本
    Yw = []  # 存放负样本
    CW_ratio = []  # 存放每轮生成的数据集中正负样本的比例，小数点后保留2位
    for i in range(epoch):  # 一共生成epoch轮数据集，通过多轮数据集的观察比较 验证正负样本的比例
        for _ in range(total_sample_num):  # 每轮生成total_sample_num个样本
            x, y = build_sample()
            X.append(x)
            if y == 1:  # 如果Y是正样本 添加进Yc列表
                Yc.append([y])
            else:  # 如果Y是负样本 添加进Yw列表
                Yw.append([y])
        a = round(float(len(Yc) / (len(Yc) + len(Yw))), 2)  # 样本的比例小数点后保留2位
        b = 1 - a
        print("第{}轮生成数据集的正负样本比为：{}:{}".format(i + 1, a, b))  # 将每轮数据集中的正负样本比例打印出来
        # print("第%d轮生成数据集的正负样本比为：%.2f:%.2f" % (i + 1, a, b))
        CW_ratio.append(str(a) + ':' + str(b))  # 将每轮的比例添加进CW_ratio列表
    print("=" * 60)
    print("%d轮数据集的正负样本比列表：%s" % (epoch, CW_ratio))  # 循环结束后 将每轮数据集正负样本的比例以列表的形式打印出来 验证是否接近1:1


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
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


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    input_size = 10  # 输入向量维度
    hidden_size1 = 270  # 第1隐藏层维度
    hidden_size2 = 20  # 第2隐藏层维度
    hidden_size3 = 4  # 第3隐藏层维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size, hidden_size1, hidden_size2, hidden_size3)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
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
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    # input_size = 5
    # model = TorchModel(input_size)
    model = TorchModel(10, 270, 20, 4)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[336., 357., 232., 201., 796., 746., 683., 785., 700., 363.],
                [823., 123., 829., 350., 574., 311., 661., 197., 466., 896.],
                [101., 663., 387., 409., 3., 548., 999., 411., 609., 18.],
                [528., 867., 525., 568., 454., 73., 175., 372., 326., 939.],
                [699., 243., 860., 412., 534., 24., 138., 439., 321., 242.],
                [216., 139., 323., 153., 158., 113., 250., 923., 467., 474.],
                [770., 84., 642., 766., 52., 501., 828., 747., 451., 544.],
                [794., 74., 887., 865., 851., 797., 273., 351., 541., 764.],
                [270., 469., 590., 773., 898., 976., 991., 576., 435., 420.],
                [707., 352., 401., 900., 212., 637., 543., 100., 902., 699.]]
    predict("model.pth", test_vec)
    # test_dataset_CW_Ratio(50000, 10)
    # print(np.around(np.random.random(10),3) * 1000)
    # print(np.around(np.random.random(100), 3).reshape(-1, 10) * 1000)
