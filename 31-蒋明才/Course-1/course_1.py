"""
 #
 # @Author: jmc-蒋明才
 # @Date: 2023/2/15 19:28
 # @Version: v1.0
 # @Description: 第1课：设计正负样本，并更改模型结构，完成模型训练
                 我设计的是一个线性回归任务 --- 共生成了16000个样本
"""
from loguru import logger
import random
import torch
import torch.nn as nn
from torch.utils.data import dataset, DataLoader
random.seed(88)


# MSELoss 均方损失
def mse_loss(pred: torch.tensor, target):
    loss = (pred - target)**2
    loss = torch.mean(loss)
    return loss


# 生成训练集
def generate_ds():
    ds = []
    for _ in range(16000):
        # 假设有4个特征
        x1 = random.random()
        x2 = random.random()
        x3 = random.random()
        x4 = random.random()
        y = 10 * (x1+x2)**2 + 25*(x3-x1) + 15*(x1+x2+x3+x4)*(x2-x4)
        ds.append([x1, x2, x3, x4, y])
    return ds


# 自定义加载数据集
class CustomDataset(dataset.Dataset):
    def __init__(self, ds: list):
        super(CustomDataset, self).__init__()
        self.ds = ds

    def __getitem__(self, item):
        cur_ds = self.ds[item]
        return torch.FloatTensor(cur_ds[:-1]), cur_ds[-1]

    def __len__(self):
        return len(self.ds)


class MyModel(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(MyModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, input_dim * 50),
            nn.Tanh(),
            nn.Linear(input_dim*50, input_dim * 50),
            nn.Tanh(),
            nn.Linear(input_dim * 50, out_dim)
        )

    def forward(self, batch_sample):
        # batch_sample (batch, input_dim)
        out = self.layer1.forward(batch_sample)  # (batch, out_dim)
        return out


class ModelTrain:
    def __init__(self):
        self.batch_size = 16
        self.epoch = 1000
        self.input_dim = 4
        self.out_dim = 1
        self.lr = 0.001

    def train(self):
        model = MyModel(self.input_dim, self.out_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        ds = CustomDataset(generate_ds())
        dataloader = DataLoader(dataset=ds, batch_size=self.batch_size, shuffle=True)
        last_loss = float('inf')
        for epoch in range(1, self.epoch + 1):
            for idx, (x, y) in enumerate(dataloader):
                pred = model.forward(x)
                loss = mse_loss(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (idx + 1) % 5 == 0:
                    if loss < last_loss:
                        logger.info(f"Epoch: {epoch}/{self.epoch}, Step: {idx}/{len(dataloader)}, Loss: {loss}")
                        torch.save(model.state_dict(), "./checkpoint/myModel.pt")
                        last_loss = loss
                if loss <= 0.001:
                    logger.info("训练完成")
                    exit(0)
        logger.info("训练完成")


if __name__ == '__main__':
    ModelTrain().train()
