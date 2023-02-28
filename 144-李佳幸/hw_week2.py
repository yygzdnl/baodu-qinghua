import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x为一个10维向量(0~999内整数），如果偶数位之和>奇数位之和，则为正样本，反之为负样本

"""

class TorchModel(nn.Module):
    def __init__(self,input_size):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size,1)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss
    
    def forward(self,x,y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_sample():
    x = np.random.randint(0,1000,size=10)
    sum_even = 0
    sum_odd = 0
    for i in range(len(x)):
        if i % 2 ==0:
            sum_even += x[i]
        else:
            sum_odd += x[i]
    if sum_even > sum_odd:
        return x,1
    else:
        return x,0

def build_dataset(sample_num):
    X = []
    Y = []
    for i in range(sample_num):
        x,y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X),torch.FloatTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    #配置参数
    epoch_num = 10
    batch_size = 20
    train_sample = 5000
    input_size = 10
    learning_rate = 0.001
    #建立模型
    model = TorchModel(input_size)
    #选择优化器
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log = []
    #获取训练集
    train_x, train_y = build_dataset(train_sample)
    #训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index*batch_size : (batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size : (batch_index+1)*batch_size]
            optim.zero_grad()
            loss = model(x,y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "mymodel.pth")
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return

def predict(model_path, input_vec):
    input_size = 10
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        sum_even = 0
        sum_odd = 0
        for i in range(len(vec)):
            if i % 2 ==0:
                sum_even += vec[i]
            else:
                sum_odd += vec[i]
        print("输入：%s, 偶数和: %s, 奇数和: %s, 预测类别：%d, 概率值：%f" % (vec, sum_even, sum_odd, round(float(res)), res))

if __name__ == "__main__":
    main()
    test_vec = []
    for i in range(5):
        test_vec.append(np.random.randint(0,1000,size=10))
    predict("mymodel.pth", test_vec)