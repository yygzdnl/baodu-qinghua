"""
规律：给定一个六维全排列，如果偶排列或自然排列则为正样本，奇排列则为负样本
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 模型
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3):
        super(TorchModel, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size_1)
        self.linear_2 = nn.Linear(hidden_size_1, hidden_size_1)
        self.linear_3 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear_4 = nn.Linear(hidden_size_2, hidden_size_2)
        self.linear_5 = nn.Linear(hidden_size_2, hidden_size_3)
        self.linear_6 = nn.Linear(hidden_size_3, hidden_size_3)
        self.linear_7 = nn.Linear(hidden_size_3, 1)
        self.activation_1 = torch.relu
        self.activation_2 = torch.sigmoid
        # self.loss = nn.functional.cross_entropy
        # self.activation_3 = torch.softmax
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        # 多层线性层多次嵌套激活函数
        x = self.linear_1(x)
        x = self.activation_1(x)
        x = self.linear_2(x)
        x = self.activation_2(x)
        x = self.linear_3(x)
        x = self.activation_1(x)
        x = self.linear_4(x)
        x = self.activation_2(x)
        x = self.linear_5(x)
        x = self.activation_1(x)
        x = self.linear_6(x)
        x = self.activation_2(x)
        x = self.linear_7(x)
        y_pred = self.activation_2(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 求逆序数
def inversion_num(arr):
    if len(arr) < 2:
        return 0
    count = 0
    tmp = arr[:]
    middle = round(len(arr) / 2)
    arr_l = tmp[:middle]
    arr_r = tmp[middle:]

    count += inversion_num(arr_l) + inversion_num(arr_r)
    i = p1 = p2 = 0

    while p1 < len(arr_l) and p2 < len(arr_r):
        if arr_l[p1] <= arr_r[p2]:
            np.append(tmp, arr_l[p1])
            p1 += 1
        else:
            np.append(tmp, arr_r[p2])
            # 计算逆序
            count = count + len(arr_l) - p1
            p2 += 1
        i += 1

    while p1 < len(arr_l):
        np.append(tmp, arr_l[p1])
        i += 1
        p1 += 1
    while p2 < len(arr_r):
        np.append(tmp, arr_r[p2])
        i += 1
        p2 += 1
    return count


def build_sample():
    x = np.random.random(6)
    if inversion_num(x) % 2:
        return x, 0
    else:
        return x, 1


def build_dataset(total):
    data = []
    types = []
    for i in range(total):
        x, y = build_sample()
        data.append(x)
        types.append([y])
    data, types = np.array(data), np.array(types)
    return torch.FloatTensor(data), torch.FloatTensor(types)


def test_eval(model, epoch, test_sample_num=100):
    model.eval()
    x, y = build_dataset(test_sample_num)
    x, y = x.cuda(), y.cuda()

    print("第%d次预测共有%d个偶排列样本， %d个奇排列样本" % (epoch, test_sample_num - sum(y), sum(y)))

    correct = wrong = 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct/test_sample_num))
    return correct/test_sample_num


def main():
    epoch_num = 600
    batch_size = 2200
    train_sample_num = 1320000
    input_size = 6
    learning_rate = 0.001
    model = TorchModel(input_size, 10, 12, 16)
    model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample_num)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample_num // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            # 运算时将数据写入gpu，相比一次性全部写入gpu更快
            x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append((loss.item()))
        print("第%d轮平均loss: %f" % (epoch + 1, np.mean(watch_loss)))
        acc = test_eval(model, epoch + 1)
        log.append([acc, float(np.mean(watch_loss))])
    # 模型数据
    torch.save(model.state_dict(), "inversion_num.mdl")
    print(log)

    plt.plot(range(len(log)), [l[0] for l in log], label='acc')
    plt.plot(range(len(log)), [l[1] for l in log], label='loss')
    plt.legend
    plt.show()
    return


def predict(model_path, input_vec):
    input_size = 6
    model = TorchModel(input_size, 10, 12, 16)
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        tensor_vec = torch.FloatTensor(input_vec).cuda()
        result = model.forward(tensor_vec)
    for vec, res in zip(input_vec, result):
        # 逆序不能直观看到，增加展示实际类型
        # 偶排列 0 < 0.5 奇排列 1 > 0.5
        print("输入：%s， 预测类别：%d， 实际类别： %d, 概率值：%f" % (vec, round(float(res)), (inversion_num(vec) + 1) % 2, res))


if __name__ == "__main__":
    main()
    # test_vec = [[0.76576114, 0.57459376, 0.94728264, 0.82396364, 0.22332013, 0.04138128],
    #             [0.68765799, 0.61846809, 0.90034621, 0.2210504,  0.46672847, 0.06901],
    #             [0.57592681, 0.48413061, 0.75094302, 0.86853686, 0.54287032, 0.84566331],
    #             [0.55447409, 0.54599965, 0.40802344, 0.47409431, 0.54273742, 0.36826861],
    #             [0.55266074, 0.69730182, 0.09508813, 0.79347399, 0.68527406, 0.62150557],
    #             [0.84567972, 0.4694959, 0.62095521, 0.56508026, 0.13724334, 0.27800808],
    #             [0.13575669, 0.39164686, 0.73885416, 0.33478729, 0.73209308, 0.87398829],
    #             [0.9908431, 0.3217038, 0.40070486, 0.69881369, 0.27276603, 0.67018808],
    #             [0.65603412, 0.29701979, 0.9053151, 0.09476201, 0.18452405, 0.11975454],
    #             [0.94002973, 0.08486079, 0.1048641, 0.29383261, 0.66165181, 0.54211266],
    #             [0.05163828, 0.47998272, 0.49362249, 0.47422207, 0.90977051, 0.12786885],
    #             [0.05050973, 0.95308079, 0.10270426, 0.01351529, 0.00207654, 0.28766321],
    #             [0.29547929, 0.10973368, 0.14683554, 0.12832415, 0.92552258, 0.41322409],
    #             [0.91821933, 0.03545748, 0.40739311, 0.14186241, 0.38106825, 0.64238214],
    #             [0.76108567, 0.7009679,  0.82916237, 0.10487043, 0.25835632, 0.69722977]]
    # predict("inversion_num.mdl", test_vec)
