import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random,json

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

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index   #每个字对应一个序号
    vocab['unk'] = len(vocab)
    with open('vocab.json', 'w') as fp:
        json.dump(vocab, fp)
    return vocab


#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    if set("abc") & set(x) and not set("df") & set(x):
        y = 1
    #指定字都未出现，则为负样本
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(vocab, sentence_length, sample_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

class NeuralNet(nn.Module):

    def __init__(self, vocab, sentence_length, vector_dim):
        super(NeuralNet, self).__init__()
        self.vocab = vocab
        self.sentence_length = sentence_length
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # embedding层
        self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        self.classify = nn.Linear(vector_dim, 1)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

    def forward(self, x):
        x = self.embedding(x)
        x = self.pool(x.transpose(1, 2)).squeeze()
        x = self.classify(x)
        y_pre = self.activation(x)
        return y_pre

    @timer
    def do_training(self, x, y, batch_size, learning_rate):
        self.train()
        #optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        log=[]
        for epoch in range(1000):
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

        torch.save(self.state_dict(), "lesson3.weight")
        return log

    def evaluate(self):
        X, Y = build_dataset(self.vocab, self.sentence_length, sample_length=20)
        print("本次预测集中共有%d个正样本，%d个负样本" % (sum(Y), sample_length - sum(Y)))
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

    def predict(self, model_path, vocab_path, input_strings):
        self.eval()
        vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
        self.load_state_dict(torch.load(model_path))
        x = []
        for input_string in input_strings:
            x.append([vocab[char] for char in input_string])  # 将输入序列化
        with torch.no_grad():
            y_pre = self.forward(torch.LongTensor(x))
        for i, input_string in enumerate(input_strings):
            print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(y_pre[i])), float(y_pre[i])))  # 打印结果
        return  y_pre

def draw(log=None):
    import matplotlib
    #import matplotlib.pyplot as plt
    from matplotlib import pyplot as plt
    matplotlib.use('TkAgg')
    print(log)
    plt.figure(1)
    plt.title("acc")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    
    plt.legend()
    plt.show()
    print("show")
    return


if __name__ == "__main__":
    #epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
    #train_sample = 500    #每轮训练总共训练的样本总数
    vector_dim = 5         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    sample_length = 200 #生成样本数据数量
    vocab = build_vocab()
    model = NeuralNet(vocab=vocab, sentence_length = sentence_length, vector_dim=vector_dim)
    X, Y = build_dataset(vocab, sentence_length, sample_length)
    log = model.do_training(X, Y, batch_size=batch_size, learning_rate=learning_rate)
    test_strings = ["ffvfee", "wwsdfg", "rqwdbg", "nakwww"]
    model.predict("lesson3.weight", "vocab.json", test_strings)
    draw(log)
