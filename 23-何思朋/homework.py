import torch
import torch.nn as nn
import  numpy as np
import  torch.nn.functional
# 定义一个模型，输入三维向量，要求第一维度和第二维相加大于第三维的是正样本，否则为负样本


# 生成数据集
def bulid_data(sample_num):
    x=torch.normal(0,5,(sample_num,3))
    y=torch.zeros((sample_num,1))
    for i in range(len(x)):
      if(x[i,0]+x[i,1]>x[i,2]):
          y[i]=1
      else:
          y[i]=0
    return torch.FloatTensor(x),torch.FloatTensor(y)




# 定义模型类
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.layer=nn.Linear(3,1)
        self.activation=torch.sigmoid
        self.loss=nn.functional.mse_loss

    def forward(self,x, y=None):
        x1=self.layer(x)
        y_pred=self.activation(x1)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

m=model()
x=torch.FloatTensor([1,2,3])
x=m.layer(x)
print(m.activation(x))

# 测试
def evaluate(model):
    model.eval()
    correct,wrong=0,0
    sample_num=1000
    x,y=bulid_data(sample_num)
    print("有%d个正样本和%d个负样本"%(sum(y),(sample_num-sum(y))))
    y_pred=model(x)
    po=0
    with torch.no_grad():
     for y1,y2 in zip(y_pred,y):
        if y1>0.5 and y2==1:
            po+=1
        elif y1<=0.5 and y2==0:
            po+=1
    print("正确率 %f" %(po/sample_num))
    return  po/sample_num


# 训练
def main():

 epoches=5
 sample_num=1000
 x,y=bulid_data(sample_num)
 batch_size=10 # 每轮训练的样本数量

 train_model=model()
 trainer=torch.optim.Adam(train_model.parameters(),lr=0.003)
 for i in range(epoches):
     train_model.train()
     watch_loss=[]
     for batch_index in range(sample_num // batch_size):
         x1 = x[batch_index * batch_size: (batch_index + 1) * batch_size]
         y1 = y[batch_index * batch_size: (batch_index + 1) * batch_size]
         trainer.zero_grad()
         loss = train_model(x1, y1)
         loss.backward()
         trainer.step()
         watch_loss.append(loss.item())
     print("=========\n第%d轮平均loss:%f" % (i + 1, float(np.mean(watch_loss))))
     acc = evaluate(train_model)

if __name__ == '__main__':
    main()