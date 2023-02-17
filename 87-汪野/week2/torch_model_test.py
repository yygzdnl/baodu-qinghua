import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

'''
规律1：x 是一个10维向量，如果存在前两个数之和小于第三个数，则为正样本，反之则为反样本
规律2：“第一个数加第二个数大于第三个数加第四个数”为正样本，其余为负样本
规律3：“第一个数的平方小于第二个数的开方”为正样本
规律4：“第一个大于0.3且第二个小于0.6”为正样本

问题1：对于规律1，随机获取的样本不均匀
解决：对数据进行筛选，保证正负样本均衡
问题2：损失函数始终无法下降，导致模型的预测效果极不准确

'''

#构建模型结构  (1 x 10) -->(10 x 7) ==> (1 x 7) x (7 x 3) ==> (1 x 3) x (3 x 1) ==> (1, 1)
class TorchModule(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(TorchModule, self).__init__()  #继承
        self.linear1 = nn.Linear(input_size,hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, 1)
        self.activation = torch.sigmoid  # 归一化
        self.loss = nn.functional.mse_loss

    #前向计算
    def forward(self, x, y = None):
        x1 = self.linear1(x)
        y1 = self.activation(x1)
        x2 = self.linear2(y1)
        y2 = self.activation(x2)
        x3 = self.linear3(y2)
        y_pred = self.activation(x3)
        #y_pred = y_pred.squeeze(-1)
        #预测的batch维度与真实的batch维度不同，按照提示需要统一维度，用squeeze将预测维度从（64，1）压缩为（64)
        if y is not None:
            return self.loss(y_pred, y) # 通过预测值和真实值计算损失
        else:
            return y_pred   #输出预测值

#生成一个样本
def build_samp():
    x = np.random.random(10)
    flag = 0
    for i in range(10):
        if i == 8:
            break
        elif x[i]+x[i+1] <= x[i+2]:
            flag = 1
            break
        else:
            flag = 0
    if flag == 1:
        return x, 1
    else:
        return x, 0
# print(build_samp())
#规律二:第一个数加第二个数大于第三个数加第四个数, 生成一个样本
def build_samp1():
    x = np.random.random(10)
    if x[0] + x[1] > x[2] + x[3]:
        return x, 1
    else:
        return x, 0

#随机生成一些样本,同时确保正负样本的数量保持均衡
def build_datasets(total_sample_num):
    x_p = []
    y_p = []  #保存标签
    X = []
    Y = []
    count = 0
    for i in range(total_sample_num):
        x, y = build_samp()
        x_p.append(x.tolist())
        y_p.append(y)

    for i, j in zip(x_p, y_p):
        #[0.9272641444096247, 0.67828606628173, 0.43697810444353347, 0.19384947955869059, 0.9449059442633664, 0.08055625756103846, 0.5588748424897552, 0.5444070359344185, 0.6540751207091766, 0.1476799423091114] 1
        if j == 0:
            X.append(i)
            Y.append(j)
        elif j == 1 and count <= (total_sample_num - sum(y_p)):
            count += 1
            X.append(i)
            Y.append(j)
        else:
            continue

    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))

#规律二:第一个数加第二个数大于第三个数加第四个数,批量生成样本
def build_datasets1(total_sample_num):
    X = []  # 数据
    Y = []  # 标签
    for i in range(total_sample_num):
        x, y = build_samp1()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))
# print(build_datasets(10))

#测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_sum = 100
    x, y = build_datasets1(test_sample_sum)  #构建测试集
    print("本次测试集中共有%d个正样本，%d个负样本" % (sum(y), (test_sample_sum - sum(y))))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  #模型预测
        for y_p, y_t in zip(y_pred, y):
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  #负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  #正样本判断正确
            else:
                wrong += 1
    print("正确预测的个数为：%d， 正确率为：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)
#训练
def train():
    #配置参数
    epoch_num = 10  #训练轮数
    batch_size = 20  #每次训练的样本个数
    train_sample = 5000  #训练的样本总数
    input_size = 10  #输入维度
    hidden_size1 = 7
    hidden_size2 = 3
    learning_rate = 0.0001  #学习率

    #构造模型
    model = TorchModule(input_size, hidden_size1, hidden_size2)

    #选择优化器
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

    #构造训练集
    train_x, train_y = build_datasets1(train_sample)  #实际的训练样本总数为len(train_y)
    log = []
    #开始训练
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = []

        for batch_index in range(len(train_y) // batch_size):   #重新计算训练样本数
            x = train_x[batch_size * batch_index : (batch_index + 1) * batch_size]
            y = train_y[batch_size * batch_index : (batch_index + 1) * batch_size]
            optim.zero_grad() #梯度归零
            loss = model(x, y) #计算损失
            loss.backward() #计算梯度
            optim.step() #更新权重
            epoch_loss.append(loss.item())
        print("-----\n第%d轮平均loss:%f" % (epoch + 1, np.mean(epoch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(epoch_loss))])
    torch.save(model.state_dict(), "model1.pth")
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
# train()

#利用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 10
    hidden_size1 = 7
    hidden_size2 = 3
    model = TorchModule(input_size, hidden_size1, hidden_size2)
    model.load_state_dict(torch.load(model_path))  #加载训练好的模型

    model.eval() #测试模型
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  #模型预测
    for vec, res in zip(input_vec, result):
        print("输入： %s, 预测类别：%d, 概率：%f" % (vec, round(float(res)), res))  #round(x)返回浮点数x的四舍五入值

if __name__ == "__main__":
    train()
    # test_x, test_y = build_datasets(10)
    # print(test_x)
    test_vec = [
        [0.6841, 0.1131, 0.6134, 0.5182, 0.7106, 0.1632, 0.7124, 0.7506, 0.6839, 0.1922],
        [0.7199, 0.0950, 0.8085, 0.3703, 0.1307, 0.5927, 0.1018, 0.2632, 0.7609, 0.8318],
        [0.2838, 0.2753, 0.2226, 0.8311, 0.7707, 0.6831, 0.9264, 0.6008, 0.9662, 0.3510],
        [0.6609, 0.7114, 0.5952, 0.5743, 0.9880, 0.1896, 0.8695, 0.2231, 0.9650, 0.9515],
        [0.4518, 0.0805, 0.6878, 0.3393, 0.8117, 0.6409, 0.8558, 0.6559, 0.2760, 0.1460],
        [0.8245, 0.6014, 0.1923, 0.0809, 0.7181, 0.3918, 0.3690, 0.0298, 0.6836, 0.0802],
        [0.2088, 0.2149, 0.6948, 0.1353, 0.4707, 0.2523, 0.4300, 0.5248, 0.6035, 0.7497],
        [0.6671, 0.5894, 0.2306, 0.6740, 0.4216, 0.8070, 0.4620, 0.8638, 0.6309, 0.0559],
        [0.2472, 0.6375, 0.7724, 0.5555, 0.4280, 0.8756, 0.1256, 0.4429, 0.2221, 0.2731],
        [0.8048, 0.8030, 0.2482, 0.2154, 0.2710, 0.4997, 0.9413, 0.9959, 0.2749, 0.7635]]

    predict("model1.pth", test_vec)
