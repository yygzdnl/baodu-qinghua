import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as  plt


#规律：输入一个十维数组，若第偶数个的均方和>第奇数个的均方和，则为正样本，否则为负样本

class MyTorchModel(nn.Module):
    def __init__(self,input_size):
        super(MyTorchModel, self).__init__()
        #self.linear=nn.Linear(input_size,8) #线性层
        #self.linear = nn.Linear(input_size, 4)  # 线性层
        self.linear = nn.Linear(input_size, 1)  # 线性层
        self.activation=torch.sigmoid #sigmod
        self.loss=nn.functional.mse_loss #loss函数采用均方差损失

    def forward(self,x,y=None):
        x=self.linear(x)
        y_pred=self.activation(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred


def build_sample():
    x=np.random.random(10)
    if (x[0]*x[0]+x[2]*x[2]+x[4]*x[4]+x[6]*x[6]+x[8]*x[8])/5 > (x[1]*x[1]+x[3]*x[3]+x[5]*x[5]+x[7]*x[7]+x[9]*x[9])/5:
        return x,1
    else:
        return x,0


def build_dataset(total_sample_num):
    X=[]
    Y=[]
    for i in range(total_sample_num):
        x,y=build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X),torch.FloatTensor(Y)


def evaluate(model):
    model.eval()
    test_sample_num=100
    x,y=build_dataset(test_sample_num)

    print("本次预测集正样本：%d,负样本：%d" %(sum(y),test_sample_num-sum(y)))

    correct,wrong=0,0
    with torch.no_grad():#使用模型预测，无需计算梯度
        y_pred=model(x)#预测
        for y_p,y_t in zip(y_pred,y):
            if float(y_p)<0.5 and int(y_t)==0:#负样本判断正确
                correct+=1
            elif float(y_p)>=0.5 and int(y_t)==1:#正样本判断正确
                correct+=1
            else:
                wrong+=1

    print("正确个数：%d，正确率：%f" % (correct,correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    #配置参数
    epoch=50
    batch_size=20
    train_sample=20000
    input_size=10 #十维向量
    lr=0.005

    model=MyTorchModel(input_size)#模型
    optim=torch.optim.Adam(model.parameters(),lr)#优化器
    log=[]

    train_x,train_y=build_dataset(train_sample)

    for epo in range(epoch):
        model.train()
        watch_loss=[]
        for batch_index in range(train_sample//batch_size):#训练一个batch
            x=train_x[batch_index*batch_size : (batch_index+1)*batch_size]
            y=train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            optim.zero_grad()
            loss=model(x,y)#计算loss
            loss.backward()#计算梯度
            optim.step()#根据梯度更新权重
            watch_loss.append(loss.item())#不理解

        print("-----------\n第%d轮平均loss:%f" % (epo+1,np.mean(watch_loss)))
        acc=evaluate(model)
        log.append([acc,float(np.mean(watch_loss))])

    torch.save(model.state_dict(),"model1.pth")

    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

def predict(model_path,input_vec):
    input_size=10
    model=MyTorchModel(input_size)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        result=model.forward(torch.FloatTensor(input_vec))
    for vec,res in zip(input_vec,result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果

if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843,0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.99871392,0.1349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894],
                [1,0.2,3,0.4,5,0.6,7,0.8,9,1]]
    predict("model1.pth", test_vec)




