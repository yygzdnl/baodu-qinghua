# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import copy
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本是纯字母文本，还是纯数字文本，还是字母和数字混合文本
y==0表示纯字母文本；y==1表示纯数字文本；y==2表示字母数字混合文本
激活函数采用ReLU，损失函数采用交叉熵损失函数cross_entropy

"""

torch.manual_seed(0)  # 设置随机种子，保证每次生成的随机数一致


class TorchModel(nn.Module):
    def __init__(self, vector_dim, hidden_size, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # embedding层
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True)  # RNN层
        self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        self.classify = nn.Linear(hidden_size, 3)  # 线性层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活层
        self.dropout = nn.Dropout(0.5)  # Dropout层
        self.bn = nn.BatchNorm1d(3)  # BN层
        self.loss = nn.functional.cross_entropy  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值；预测值为一组概率分布的列表构成的tensor，因为是多分类
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_size)
        x = self.pool(x.transpose(1, 2)).squeeze()  # (batch_size, sen_len, hidden_size) -> (batch_size, hidden_size)
        x = self.classify(x)  # (batch_size, hidden_size) -> (batch_size, 3)
        x = self.relu(x)  # (batch_size, 3) -> (batch_size, 3)
        x = self.dropout(x)  # (batch_size, 3) -> (batch_size, 3)
        y_pred = self.bn(x)  # (batch_size, 3) -> (batch_size, 3)
        # print(y_pred.shape, "y_pred.shape:")
        # if y is not None:
        #     print(y.shape, "y.shape:")
        if y is not None:
            # return self.loss(y_pred, y())  # 预测值和真实值计算损失
            return self.loss(y_pred, y.squeeze())  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果，是一个多分类标签，列表


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"  # 字符集：字母加数字
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 样本分为3种类型：纯字母文本，纯数字文本，字母数字混合文本
# 标签y==0：纯字母文本
# y==1：纯数字文本
# y==2：字母数字混合文本
# 为了保证生成的样本中3种类型比例一致，采用if判断手动生成样本
def build_sample(i, vocab, sentence_length):
    # 根据整数i%3的结果依次生成3类样本
    # 生成纯字母文本，y的标签为0
    if i == 0:
        x = [random.choice(list("abcdefghijklmnopqrstuvwxyz")) for _ in range(sentence_length)]
        y = 0
    # 生成纯数字文本，y的标签为1
    elif i == 1:
        x = [random.choice(list("0123456789")) for _ in range(sentence_length)]
        y = 1
    # 生成字母数字混合文本，y的标签为2
    else:
        x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
        y = 2
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少，生成的3类样本比为1:1:1
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        if i % 3 == 0:
            x, y = build_sample(0, vocab, sentence_length)
            dataset_x.append(x)
            # dataset_y.append(y)
            dataset_y.append([y])
        elif i % 3 == 1:
            x, y = build_sample(1, vocab, sentence_length)
            dataset_x.append(x)
            # dataset_y.append(y)
            dataset_y.append([y])
        else:
            x, y = build_sample(2, vocab, sentence_length)
            dataset_x.append(x)
            # dataset_y.append(y)
            dataset_y.append([y])

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# test 用于测试生成的数据集中3类样本的比例是否接近1:1:1，epoch是测试的轮次，
# 最后将epoch轮生成的数据集中3类样本的比例以列表的形式打印出来 以验证样本比是否接近1:1:1
def test_dataset_CLASS_Ratio(total_sample_num, vocab, sentence_length, epoch):
    X = []  # 存放生成的x向量
    Y0 = []  # 存放纯字母文本样本
    Y1 = []  # 存放纯数字文本样本
    Y2 = []  # 存放混合文本样本
    CW_ratio = []  # 存放每轮生成的数据集中3类样本的比例，小数点后保留2位
    for i in range(epoch):  # 一共生成epoch轮数据集，通过多轮数据集的观察比较 验证3类样本的比例
        for _ in range(total_sample_num):  # 每轮生成total_sample_num个样本
            # 每次根据_ % 3的结果，自动生成对样的样本
            x, y = build_sample(_ % 3, vocab, sentence_length)
            X.append(x)
            if y == 0:  # 如果y是纯字母样本 添加进Y0列表
                Y0.append([y])
            elif y == 1:  # 如果y是纯数字样本 添加进Y1列表
                Y1.append([y])
            else:  # 如果Y是字母数字混合样本 添加进Y2列表
                Y2.append([y])
        a = float(len(Y0) / (len(Y0) + len(Y1) + len(Y2)))  # 样本的比例小数点后保留2位
        b = float(len(Y1) / (len(Y0) + len(Y1) + len(Y2)))
        c = 1 - a - b
        # print("第{}轮生成数据集的正负样本比为：{}:{}".format(i + 1, a, b))  # 将每轮数据集中的正负样本比例打印出来
        print("第%d轮生成数据集的三类样本比为：%.2f:%.2f:%.2f" % (i + 1, a, b, c))
        CW_ratio.append(str(round(a, 2)) + ':' + str(round(b, 2)) + ':' + str(round(c, 2)))  # 将每轮的比例添加进CW_ratio列表
    print("=" * 60)
    print("%d轮数据集的正负样本比列表：%s" % (epoch, CW_ratio))  # 循环结束后 将每轮数据集正负样本的比例以列表的形式打印出来 验证是否接近1:1:1


# 建立模型
def build_model(vocab, char_dim, hidden_size, sentence_length):
    model = TorchModel(char_dim, hidden_size, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(300, vocab, sample_length)  # 建立300个用于测试的样本
    print("本次预测集中共有%d个纯字母样本，%d个纯数字样本，%d个字母数字混合样本" % (
        sum(y.squeeze().numpy() == 0), sum(y.squeeze().numpy() == 1), sum(y.squeeze().numpy() == 2)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if np.argmax(y_p.numpy()) == y_t.numpy():
                correct += 1  # 样本判断正确 当预测值多分类标签最大值的索引与真实标签的值相同时 预测正确，即样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 6000  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    hidden_size = 10  # RNN层隐向量神经元个数
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, hidden_size, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    hidden_size = 10  # RNN层隐向量神经元个数
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, hidden_size, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
        print(result)
    for i, input_string in enumerate(input_strings):
        str = ""
        if np.argmax(result.numpy()[i]) == 0:
            str = "纯字母样本"
        elif np.argmax(result.numpy()[i]) == 1:
            str = "纯数字样本"
        else:
            str = "字母数字混合样本"
        print("输入：%s, 预测类别：%d, %s,概率值：%f" % (
            input_string, np.argmax(result.numpy()[i]), str, np.max(torch.softmax(result, dim=1).numpy()[i])))  # 打印结果
    print(torch.softmax(result, dim=1))


if __name__ == "__main__":
    # main()
    # test_strings = ["ffvfee", "wwsdfg", "rqwdbg", "nakwww"]
    test_strings = ["ffvfee", "111122", "rqw234", "nakwww"]
    predict("model.pth", "vocab.json", test_strings)
    # test_dataset_CLASS_Ratio(6000, build_vocab(), 20, 10)
