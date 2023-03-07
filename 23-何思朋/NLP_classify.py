import json

import numpy as np
import  torch.nn as nn
import  torch

import random
import matplotlib.pyplot as plt
# 文本分类任务

'''
一个文本分类的任务
判断文本中是否出现给定字符

'''

class NLPmodel(nn.Module):
    def __init__(self,string_length,vec_num,voc):
        super(NLPmodel, self).__init__()
        self.embedding=nn.Embedding(len(voc),vec_num)
        self.pool=nn.AvgPool1d(string_length)
        self.linear=nn.Linear(vec_num,1)
        self.activation=torch.sigmoid
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        x=self.embedding(x)
        x = self.pool(x.transpose(1, 2)).squeeze()
        x=self.linear(x)
        y_pred=self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 建立数字表
def build_voc():
    char='123456789abcdefghijklmnopqrstuvwxyz'
    voc={}
    for index,chari in enumerate(char):
        voc[chari]=index
    voc['unk'] = len(voc)
    return voc


# 随机生成一个样本
def build_sample(voc,string_length):

    x = [random.choice(list(voc.keys())) for i in range(string_length)]
    if set("1ax") & set(x):
        y = 1
    else:
        y = 0
    x = [voc.get(word, voc['unk']) for word in x]
    return x, y

# 生成数据集
def build_dataset(sample_num,string_length,voc):
     x=[]
     y=[]

     for i in range(sample_num):
         x1,y1=build_sample(voc,string_length)
         x.append(x1)
         y.append([y1])
     return  torch.LongTensor(x), torch.FloatTensor(y)


# 测试
def evaulate(model,string_length,voc):
    sample_num=250
    model.eval()
    x,y=build_dataset(sample_num,string_length,voc)
    print("正样本个数%d，负样本个数%d"%(sum(y),len(y)-sum(y)))
    correct,wrong=0,0
    with torch.no_grad():
        y_pred=model(x)
        for y1,y2 in zip(y_pred,y):
            if float(y1)>=0.5 and int(y2)==1:
                correct+=1
            elif float(y1)<0.5 and int(y2)==0:
                correct+=1
            else:
                wrong+=1
    print("正确率为%f"%(correct/(correct+wrong)))
    return correct / (correct + wrong)

# 训练
def main():
    epoch_num=20
    batch_size=20
    sample_train=400
    string_length=5
    vec_dim=3
    learing_rate=0.03
    voc=build_voc()
    model=NLPmodel(string_length,vec_dim,voc)
    optisim=torch.optim.Adam(model.parameters(),lr=learing_rate)

    log=[]

    for i in range(epoch_num):
        model.train()
        watch_loss=[]
        for batch in range(int(sample_train / batch_size)):
             x, y = build_dataset(batch_size,string_length,voc)
             optisim.zero_grad()
             loss=model(x,y)
             loss.backward()
             optisim.step()

             watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (i + 1, np.mean(watch_loss)))
        r1=evaulate(model,string_length,voc)
        log.append([r1, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    return

if __name__ == '__main__':
    main()