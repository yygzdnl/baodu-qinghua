import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class TestTorch(nn.Module):
    def __init__(self,input_size):
        super(TestTorch,self).__init__()
        self.linear = nn.Linear(input_size,1)   #线性层
        # self.activation = torch.sigmoid         #激活函数
        # self.loss = nn.functional.mse_loss  # 损失函数
        self.activation = torch.sigmoid
        self.loss = nn.functional.l1_loss


    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self,x,y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)         #预测值和真实值计算损失
        else:
            return y_pred                       #输出预测结果


def build_train_sample():
    x = np.random.random(5)
    if x[0] > x[1]:
        return x,1
    else:
        return x,0

# 构建训练样本
def build_dataset(total_sample):
    X,Y = [],[]
    for i in range(total_sample):
        x,y = build_train_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))


# 测试模型
def evaluate(model):
    model.eval()        #进入测试模式
    test_sample_num = 200
    x,y = build_dataset(test_sample_num)    #测试样本
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))

    correct, wrong = 0, 0
    TP,TN,FP,FN = 0,0,0,0
    with torch.no_grad():
        y_pred = model(x)       #预测值
        for y_p,y_true in zip(y_pred,y):
            if int(y_true) == 1 and float(y_p) > 0.5:
                correct += 1
                TP += 1
            elif int(y_true) == 0 and float(y_p) < 0.5:
                correct += 1
                TN += 1
            elif int(y_true) == 1 and float(y_p) <= 0.5:
                wrong += 1
                FN += 1
            else:
                wrong += 1
                FP += 1

    totalP = TP + FP if TP + FP > 0 else 0.1
    totalT = TP + FN if TP + FN > 0 else 0.1
    print("正确预测个数：%d, 准确率：%f，精准率：%f，召回率：%f" % (correct, correct / (correct + wrong), TP/totalP, TP/totalT))
    return correct / (correct + wrong), TP/totalP, TP/totalT


# 训练
def train():
    batch_size = 20      #每次训练样本个数
    epoch_num = 10       #训练轮数
    train_sample = 5000  #训练样本数
    input_size = 5       #输入向量维度
    learning_rate = 0.01 #学习率

    # 创建模型
    model = TestTorch(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log=[]
    # 构建训练样本
    train_x,train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()       #此处，因每轮都会进入测试模型，故每轮训练都重新进入训练模式
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y_true = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            optim.zero_grad()           # 梯度归零
            loss = model(x, y_true)     # 每个批次的损失函数
            loss.backward()             # 计算梯度，反向传播
            optim.step()                # 更新权重
            watch_loss.append(loss.item())
        # for x,y_true in zip(train_x,train_y):
        #     loss = model(x,y_true)
        #     loss.backward()
        #     optim.step()
        #     watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        acc,pre,recall = evaluate(model)  # 测试本轮模型结果
        log.append([acc, pre, recall, float(np.mean(watch_loss))])

    # 保存模型参数
    torch.save(model.state_dict(), "model.pth")

    # 画图
    # print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="accuracy")  # 画accuracy曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="precision")  # 画precision曲线
    plt.plot(range(len(log)), [l[2] for l in log], label="recall")  # 画recall曲线
    plt.plot(range(len(log)), [l[3] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 用训练好的模型做预测验证
def predict(model_path, input_vec):
    input_size = 5
    model = TestTorch(input_size)
    model.load_state_dict(torch.load(model_path))   #加载训练好的权重

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec,res in zip(input_vec,result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    # 先训练
    train()

    input_vec = np.array([
        [0.47889086, 0.47889085, 0.31082123, 0.03504317, 0.18920843],
        [1.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
        [-0.78797868, 0.67482528, -0.13625847, 0.34675372, 0.99871392],
        [0.1349776, 0.13497767, 0.92579291, 0.41567412, 0.7358894]
    ])
    predict("model.pth", input_vec)






