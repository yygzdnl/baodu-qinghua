import torch
import torch.nn as nn
import torchvision
import numpy as np

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MyModel(nn.Module):
    def __init__(self, input_size, hiden_size, output_size):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hiden_size)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(hiden_size, output_size)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        return x


class MyDataset(Dataset):
    def __init__(self, mode="train"):
        dataset = torchvision.datasets.MNIST(root='./MNIST',
                                             train=(mode == "train"),
                                             transform=torchvision.transforms.ToTensor(),
                                             download=True)
        data, target = dataset.data, dataset.targets
        self.data = data.reshape(-1, 28 * 28).float()
        self.target = torch.zeros(target.size()[0], 10)
        self.target.scatter_(1, target.reshape(-1, 1), 1)

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

    def __getitem__(self, index):
        # 打开index对应图片进行预处理后return回处理后的图片和标签
        pic = self.data[index]
        label = self.target[index]
        return pic, label


def main():
    # 超参数设置
    device = "cpu" if torch.cuda.is_available() else "GPU"
    learning_rate = 0.01
    batch_size = 128
    num_epoch = 5

    # 加载数据集
    train_data_loader = DataLoader(MyDataset(mode="train"),
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4)
    test_data_loader = DataLoader(MyDataset(mode="test"),
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=4)
    # 加载模型
    model = MyModel(28 * 28, 1000, 10)
    # 设置优化器
    opt = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # 设置损失函数
    ce_loss = torch.nn.CrossEntropyLoss()

    # 开始训练啦
    loss_list = []
    acc_list = []
    best_acc = 0
    for epoch in range(num_epoch):
        model.train()
        epoch_loss = []
        print("第【%d】轮训练" % (epoch + 1), flush=True)
        for data, target in tqdm(train_data_loader):
            opt.zero_grad()  # 梯度归零
            pred = model(data)
            loss = ce_loss(pred, target)
            loss.backward()  # 计算梯度
            opt.step()  # 更新权重
            epoch_loss.append(loss.item())
        loss_list.append(np.mean(epoch_loss))
        print("=========第【%d】轮平均loss: %.4f" % (epoch + 1, float(np.mean(epoch_loss))), flush=True)

        # 每轮训练完成后要测试一下
        model.eval()
        correct, wrong = 0, 0
        with torch.no_grad():
            print("第【%d】轮测试" % (epoch + 1), flush=True)
            for data, target in tqdm(test_data_loader):
                pred = model(data)
                if torch.argmax(pred) == torch.argmax(target):
                    correct += 1
                else:
                    wrong += 1
        acc = correct / (correct + wrong)
        print("第【%d】轮正确预测个数：%d, 正确率：%.4f" % (epoch + 1, correct, acc), flush=True)
        acc_list.append(acc)
        if acc > best_acc:
            torch.save(model.state_dict(), "model_best.pth")
            best_acc = acc
    torch.save(model.state_dict(), "model_last.pth")
    return


if __name__ == '__main__':
    main()
