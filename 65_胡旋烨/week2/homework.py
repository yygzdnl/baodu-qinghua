import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter
import time


# static Params
num_epoch = 10
learning_rate = 0.01
batch_size_train = 128
batch_size_test = 1000
gpu = torch.cuda.is_available()
# gpu = False

log_dir = r"./logs"
model_path = r"./weights"
data_path = r'./MNIST'

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_path, exist_ok=True)


class MyModel(nn.Module):
    def __init__(self, input_size, hiden_size, output_size):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hiden_size)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(hiden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x


def main():
    # 1,加载数据集
    train_dataset = torchvision.datasets.MNIST(root=data_path,
                                               train=True,
                                               transform=torchvision.transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root=data_path,
                                              train=False,
                                              transform=torchvision.transforms.ToTensor(),
                                              download=True)
    # 2,创建dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size_train,
                                  shuffle=True,
                                  num_workers=4)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size_test,
                                 shuffle=False,
                                 num_workers=4)
    train_data_size = len(train_dataloader)
    test_data_size = len(test_dataloader)
    print("训练集数量：%d" % train_data_size)
    print("测试集数量：%d" % test_data_size)

    # 3，创建模型
    model = MyModel(28 * 28, 1000, 10)
    print(model)
    if gpu:
        model.cuda()

    # 4,定义损失函数、优化器、tensorboard等
    loss_func = nn.CrossEntropyLoss()
    if gpu:
        loss_func.cuda()

    optim = opt.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, time.strftime('%Y%m%d-%H%M%S')))

    # 5,开始训练
    total_train_step = 0
    best_acc = 0
    for epoch in range(num_epoch):
        model.train()
        for data, targets in train_dataloader:
            data = data.reshape(-1, 28 * 28).float()  # [-1, 28*28]
            targets_onehot = torch.zeros(targets.size()[0], 10)
            targets_onehot.scatter_(1, targets.reshape(-1, 1), 1)
            if gpu:
                data, targets_onehot = data.cuda(), targets_onehot.cuda()
            optim.zero_grad()
            outputs = model(data)
            loss = loss_func(outputs, targets_onehot)
            loss.backward()
            optim.step()
            if total_train_step % 100 == 0 and total_train_step != 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, total_train_step, train_data_size,
                    100. * total_train_step / train_data_size, loss.item()))
            writer.add_scalar('loss', loss.item(), total_train_step)
            total_train_step += 1

        # 6，每次训练完成后需要进行测试
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data, targets in test_dataloader:
                data = data.reshape(-1, 28 * 28).float()  # [-1, 28*28]
                if gpu:
                    data, targets = data.cuda(), targets.cuda()
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        acc = correct / total
        print('Test Accuracy: {}/{} ({:.2f}%)'.format(correct, total, 100. * acc))
        writer.add_scalar('test_accuracy', acc, total_train_step)

        # 7，保存模型
        if acc > best_acc:
            torch.save(model.state_dict(), os.path.join(model_path, "model_best.pth"))
            best_acc = acc
    torch.save(model.state_dict(), os.path.join(model_path, "model_last.pth"))
    writer.close()
    return


def predict(model_path, input_vec):
    # 使用训练好的模型做预测
    model = MyModel(28 * 28, 1000, 10)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(input_vec)  # 模型预测
        _, predicted = torch.max(result.data, 1)
        print(predicted)


if __name__ == '__main__':
    main()
