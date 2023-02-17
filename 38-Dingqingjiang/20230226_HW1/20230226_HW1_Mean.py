import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个n维向量，如果向量均值大于0.5，则为正样本，反之为负样本
"""

# 平均数大于0.5为正，小于0.5为负
# n维向量样本生成器
def sample(samplesize, input_size):
    x = []
    for i in range(samplesize):
        x.append(np.random.random(input_size))
    return x
# 样本判断，0为负样本，1为正样本
def build_dataset(samplesize, input_size):
    x = sample(samplesize, input_size)
    y = []
    for i in range(len(x)):
        if np.mean(x[i]) > 0.5:
            y.append([1])
        else:
            y.append([0])
    return torch.FloatTensor(x), torch.FloatTensor(y)

# Predictive Module: Linear + Sigmoid MSE
class TorchModule(nn.Module):
    def __init__(self, input_size):
        super(TorchModule, self).__init__()
        self.Linear = nn.Linear(input_size, 1)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        x = self.Linear(x)
        y_pred = self.activation(x)
        if y is not None:
            loss = self.loss(y_pred, y)
            return loss
        else:
            return y_pred

# Model Evaluation
def evaluate(model):
    model.eval() #测试模式
    test_samplesize = 100
    x, y = build_dataset(test_samplesize, input_size)
    print('本次测试集中共有%d个正样本，%d个负样本。' % (sum(y), len(y) - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad(): #不计算梯度
        y_pred = model.forward(x)
        for y_p, y_t in zip(y_pred, y):
            # break
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1
            else:
                wrong += 1
    print('正确个数：%d，错误个数：%d.' %(correct, wrong))
    return correct/(correct + wrong)

# Run Model
def main():
    samplesize = 1000

    epoch_num = 10
    batch_size = 20
    learning_rate = 0.1
    model = TorchModule(input_size)
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    loss_log = []

    #
    train_x, train_y = build_dataset(samplesize, input_size)
    for epoch in range(epoch_num):
        model.train() # 训练模式
        loss_list = []
        for batch_index in range(samplesize // batch_size):
            x = train_x[batch_size * batch_index : batch_size * (batch_index+1)]
            y = train_y[batch_size * batch_index : batch_size * (batch_index+1)]
            optim.zero_grad() # 梯度归零
            loss = model.forward(x, y) #计算loss
            loss.backward() # 求导
            optim.step() # 更新权重
            loss_list.append(loss.item())
            # item()函数：取出单元素张量的元素值并返回该值，保持该元素类型不变。取值的精度更高
        print('========= 第%d轮平均loss: %f' %(epoch+1, np.mean(loss_list)))
        #测试本轮模型结果
        accuracy = evaluate(model)
        loss_log.append([accuracy, np.mean(loss_list)])
    #保存模型
    torch.save(model.state_dict(), "../../../八斗人工智能-清华班/第二周 深度学习基础/week2 深度学习常用组件/model1.pth")
    #画图
    print(loss_log)
    plt.plot(range(len(loss_log)), [l[0] for l in loss_log], label = 'Accuracy')
    plt.plot(range(len(loss_log)), [l[1] for l in loss_log], label = 'Batch_mean_loss')
    plt.legend()
    plt.show()
    return

# Prediction
def predict(model_path, input_vec):
    model = TorchModule(input_size)
    model.load_state_dict(torch.load(model_path)) # 加载训练好的权重
    # print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result_pred = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result_pred):
        print('=== 输入：%s, 输出：%d, 平均值：%f, 概率值：%f' % (vec, round(float(res)), sum(vec)/len(vec), res))

# Run
if __name__ == '__main__':
    # 指定n的维数
    input_size = 8
    main()
    # 生成一个测试向量
    test_vec = sample(4, input_size)
    predict('model1.pth', test_vec)







