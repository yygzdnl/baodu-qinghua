import os
import time

import torch
import torch.nn as nn
import torch.optim as opt

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

"""
设计一个nlp分类器，如果单词中含有大写字母，则分类为0，全为小写则分类为1
训练样本为哈利波特的英文数据集，地址：dataset.actical.txt
"""
vector_dim = 50
max_centence_length = 5
batch_size = 64
lr = 0.0001
num_epochs = 10
train_prob = 0.8
gpu = True
log_dir = r'./logs'
model_path = r'./weights'


def clean(word):
    stop_words = ['.', ':', '?', '!', ',', '"', "'", '-', '.', ';', '\n', '\t', '\r', '\a', '\f', '\v', '\b']
    for item in stop_words:
        if item in word:
            word.replace(item, '', 9999999999999)
    return word


def get_words(path):
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    words = []
    for line in lines:
        line = line.strip().split(' ')
        for word in line:
            word = clean(word)
            words.append(word)
    return words


def get_vocab():
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


def build_datasets(words, vocab):
    samples = []
    labels = []
    for word in words:
        sample = []
        label = []
        for chr in word:
            sample.append(vocab.get(chr, vocab['unk']))
        # 确保输出的尺寸一致，多的截掉，少的补全
        if len(sample) > max_centence_length:
            sample = sample[:max_centence_length]
        while len(sample) < max_centence_length:
            sample.append(vocab['unk'])

        if word == word.lower():
            label.append(1)
        else:
            label.append(0)
        samples.append(sample)
        labels.append(label)
    return torch.Tensor(samples), torch.Tensor(labels)


class MyDataset(Dataset):
    def __init__(self, words, vocab):
        super(MyDataset, self).__init__()
        self.samples = []
        self.labels = []
        for word in words:
            sample = []
            label = []
            for chr in word:
                sample.append(vocab.get(chr, vocab['unk']))
            # 确保输出的尺寸一致，多的截掉，少的补全
            if len(sample) > max_centence_length:
                sample = sample[:max_centence_length]
            while len(sample) < max_centence_length:
                sample.append(vocab['unk'])

            if word == word.lower():
                label.append(1)
            else:
                label.append(0)
            self.samples.append(sample)
            self.labels.append(label)

    def __getitem__(self, index):
        # 返回一个数据
        return torch.IntTensor(self.samples[index]), torch.FloatTensor(self.labels[index])

    def __len__(self):
        # 返回数据集大小
        return len(self.samples)


class MyModel(nn.Module):
    def __init__(self, vocab_dim, vector_dim, centence_length):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, vector_dim)
        self.pool = nn.AvgPool1d(centence_length)
        self.classifer = nn.Linear(vector_dim, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x：[batch_size， centence_length]
        x = self.embedding(x)  # [batch_size， centence_length, vector_dim]
        x = self.pool(x.transpose(1, 2)).squeeze()  # [batch_size, vector_dim]
        x = self.act(self.classifer(x))  # [batch_size, 1]
        return x


def main():
    words = get_words('./dataset/acticle.txt')
    vocab = get_vocab()

    # 1,构建自己的数据集
    # samples, labels = build_datasets(words, vocab)

    train_dataset = MyDataset(words[:int(train_prob * len(words))], vocab)
    test_dataset = MyDataset(words[int(train_prob * len(words)):], vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_data_size = len(train_dataloader)
    test_data_size = len(test_dataloader)
    print("训练集数量：%d" % train_data_size)
    print("测试集数量：%d" % test_data_size)

    # 2,构建自己的模型
    model = MyModel(len(vocab), vector_dim, max_centence_length)
    print(model)
    if gpu:
        model.cuda()

    # 3,定义损失函数、优化器、tensorboard等
    loss_func = nn.MSELoss()
    if gpu:
        loss_func.cuda()
    optim = opt.SGD(model.parameters(), lr=lr, momentum=0.9)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, time.strftime('%Y%m%d-%H%M%S')))

    # 5,开始训练
    total_train_step = 0
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        for data, targets in train_dataloader:
            if gpu:
                data = data.cuda()
                targets = targets.cuda()
            optim.zero_grad()
            outputs = model(data)
            loss = loss_func(outputs, targets)
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
                if gpu:
                    data = data.cuda()
                    targets = targets.cuda()
                outputs = model(data)
                predicted = outputs.data
                predicted[predicted >= 0.5] = 1
                predicted[predicted < 0.5] = 0
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
    vocab = get_vocab()

    # 将输入的单词转变为向量
    sample = []
    for chr in input_vec:
        sample.append(vocab.get(chr, vocab['unk']))
    # 确保输出的尺寸一致，多的截掉，少的补全
    if len(sample) > max_centence_length:
        sample = sample[:max_centence_length]
    while len(sample) < max_centence_length:
        sample.append(vocab['unk'])
    sample = torch.IntTensor([sample])

    # 加载数据集
    model = MyModel(len(vocab), vector_dim, max_centence_length)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(sample)  # 模型预测
        predicted = result.data
        predicted[predicted >= 0.5] = 1
        predicted[predicted < 0.5] = 0
        print(predicted)


if __name__ == '__main__':
    # main()
    predict('./weights/model_best.pth', 'Harry')  # 1
    predict('./weights/model_best.pth', 'and')  # 1
    predict('./weights/model_best.pth', 'Poter')  # 1
    # words = get_words('./dataset/acticle.txt')
    # vocab = get_vocab()
    # ds = MyDataset(words, vocab)
    # for i, word in enumerate(words):
    #     print('%s: %s' % (word, str(ds.__getitem__(i))))
    #     if i >= 20:
    #         break