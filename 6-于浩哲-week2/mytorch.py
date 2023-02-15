import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
"""
6 于浩哲
"""
"""
构建学习任务：
0.复现学习任务(已完成)
1.x是一个4维向量的矩阵，如果x0*x2+x3>4*x1^2则为正样本，反之为负样本(已完成)
2.x是一个10维向量，定义域范围为0-9，输出为这十个数中的众数 (待完成，softmax还不会用)
3.拟合数据 (待完成)
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 线性层
        self.activation = torch.sigmoid  # 归一化函数
        self.loss = nn.functional.mse_loss  # 均方差损失

    # 当输入真实标签，有监督返回loss，无真实标签，返回预测值
    def forward(self, x, y=None):
        tmp = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(tmp)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_sample():
    x = np.random.random(4)
    # 二分类函数，输出为0，1
    if x[0] * x[2] + x[3] > 4*x[1] ** 2:
        return x, 1
    else:
        return x, 0


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)


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
    epoch_num = 20  # 总轮次
    batch_size = 50  # 批次
    train_sample_num = 5000  # 每轮训练样本数
    input_size = 4  # 输入向量的大小
    learning_rate = 0.01
    # 建立模型
    model1 = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    log = []  # 损失曲线
    # 建立样本
    train_x, train_y = build_dataset(train_sample_num)
    # 开始训练
    for epoch in range(epoch_num):
        model1.train()
        watch_loss = []
        for batch_index in range(train_sample_num // batch_size):
            x = train_x[batch_size * batch_index:batch_size * (batch_index + 1)]
            y = train_y[batch_size * batch_index:batch_size * (batch_index + 1)]
            optim.zero_grad()
            loss = model1.forward(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model1)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model1.state_dict(), "model1.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    input_size = 4
    model1 = TorchModel(input_size)
    model1.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model1.state_dict())

    model1.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model1.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.47889086, 0.15229675, 0.03504317, 0.18920843],
    #             [0.94963533, 0.0, 0.0, 0.0],
    #             [0.78797868, 0.67482528, 0.13625847, 0.99871392],
    #             [0.1349776, 0.59416669, 0.92579291, 0.41567412]]
    test_vec1 = np.random.random([20, 4])
    predict("model1.pth", test_vec1)
