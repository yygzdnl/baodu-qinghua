import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

def timer(func):
    def func_wrapper(self,*args, **kwargs):
        from time import time
        time_start = time()
        result = func(self,*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result
    return func_wrapper

def get_xy(n=100, is_np=False):
    W = np.asarray([[3],[1]])
    X = np.random.random((n,2))
    Y = np.dot(X, W) + 0.5 - np.random.rand()
    Y = np.where(Y > 2, 1, 0)
    if is_np:
        return X, Y
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()


class NeuralNet(nn.Module):

    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.fc0 = nn.Linear(in_features=input_dim, out_features=2, bias=True)
        # self.fc1 = nn.Linear(in_features=10, out_features=10, bias=True) #多了不行
        # self.fc2 = nn.Linear(in_features=10, out_features=10, bias=True)
        self.fc3 = nn.Linear(in_features=2, out_features=1, bias=True)
        self.sigmoid = nn.functional.sigmoid
        #self.sigmoid = nn.functional.relu 这个不行
        self.loss = nn.functional.cross_entropy
        #self.loss = nn.functional.mse_loss 这个不行

    def forward(self, x):
        f0 = self.fc0(x)
        f0 = self.sigmoid(f0)
        # f1 = self.fc1(f0)
        # f1 = self.sigmoid(f1)
        # f2 = self.fc2(f1)
        # f2 = self.sigmoid(f2)
        f3 = self.fc3(f0)
        y_pre = self.sigmoid(f3)
        return y_pre

    @timer
    def do_training(self, x, y, batch_size=10):#20不如10好，大的不好小的好
        #x = Variable(x, requires_grad=True) #为什么我需要加这句？因为loss传错参数了
        self.train()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        log=[]
        for epoch in range(30000):
            batch_rows = np.random.randint(x.shape[0], size=batch_size)
            #print(x.shape, y.shape)
            x_batch = x[batch_rows, :]
            #y_batch = y[batch_rows].squeeze().to(dtype=torch.long) # ? error
            y_batch = y[batch_rows, :]
            optimizer.zero_grad()
            #print(x_batch.shape, y_batch.shape)
            #print(x.dtype,y.dtype)
            y_pre = self.forward(x_batch)
            #print(y_pre.shape, y_batch.shape)
            # frward + loss + backward + optimize
            loss = self.loss(y_pre.squeeze(), y_batch.squeeze())
            #print("总loss: ",loss)
            mean_loss = torch.mean(loss, dim=0)
            loss.backward()
            optimizer.step()
            print("=========\n第%d轮平均loss:%f" % (epoch + 1, float(mean_loss)))
            acc = self.evaluate()
            log.append([acc, float(mean_loss)])

        torch.save(self.state_dict(), "lesson2.weight")
#        draw(log)

    def evaluate(self, test_sample_num=10):
        X, Y = get_xy(n=test_sample_num)
        print("本次预测集中共有%d个正样本，%d个负样本" % (sum(Y), test_sample_num - sum(Y)))
        self.eval()
        correct, wrong = 0, 0
        with torch.no_grad():
            y_eval = self.forward(X)
            for y_p, y_t in zip(y_eval, Y):  # 与真实标签进行对比
                if float(y_p) < 0.5 and int(y_t) == 0:
                    correct += 1  # 负样本判断正确
                elif float(y_p) >= 0.5 and int(y_t) == 1:
                    correct += 1  # 正样本判断正确
                else:
                    wrong += 1
            loss = torch.mean(self.loss(y_eval, Y), dim=0)
            print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
            return correct / (correct + wrong)

    def predict(self, x, model_path):
        self.eval()
        self.load_state_dict(torch.load(model_path))
        with torch.no_grad():
            y_pre = self.forward(x)
        for x, y in zip(x, y_pre):
            print("输入：%s, 预测类别：%d, 概率值：%f" % (x, round(float(y)), y))  # 打印结果
        return  y_pre

def draw(log):
    import matplotlib.pyplot as plt
    print(log)
    plt.figure(1)
    plt.title("acc")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

def draw2d():
    import matplotlib
    from matplotlib import pyplot as plt
    plt.figure(2)

    matplotlib.use('TkAgg')

    X, Y = get_xy(is_np=True)

    plt.subplot(1, 2, 1)
    plt.title("y_true")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(X[np.argwhere(Y == 1),0],X[np.argwhere(Y == 1),1], color='red')
    plt.scatter(X[np.argwhere(Y == 0),0],X[np.argwhere(Y == 0),1], color='blue')

    model = NeuralNet(X.shape[-1])
    y_pre = model.predict(torch.from_numpy(X).float(), "lesson2.weight")
    # for layer, param in model.state_dict().items():
    #     print(layer, param)
    #print(model.state_dict()["fc0.weight"].numpy())
    
    plt.subplot(1, 2, 2)
    plt.title("y_pre")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(X[np.argwhere(y_pre >= 0.5), 0], X[np.argwhere(y_pre >= 0.5), 1], color='red')
    plt.scatter(X[np.argwhere(y_pre < 0.5), 0], X[np.argwhere(y_pre < 0.5), 1], color='blue')
    # x = np.linspace(0, 5, 50)
    # y = model.state_dict()["fc.weight"].numpy() * x + model.state_dict()["fc.bias"].numpy()
    #plt.plot(x, y, color="gray", linewidth=1)

    plt.suptitle("compare")
    plt.show()

if __name__ == "__main__":
    X, Y = get_xy()
    model = NeuralNet(X.shape[-1])
    model.do_training(X, Y)

    draw2d()
