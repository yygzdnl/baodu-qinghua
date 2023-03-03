#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
"""
基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
对文本类型进行分类，分类包括：教育、健康、动物、娱乐、游戏、体育、犯罪和其它
"""
nlp_test_string = \
    [
        "教育是国家发展的第一动力",
        "在动物园可以看见熊猫吗？",
        "为什么要犯罪",
        "为什么需要重视健康",
        "我要去体育馆",
        "我喜欢喝茶",
        "我喜欢玩王者游戏",
        "你喜欢出去娱乐吗",

        "王者游戏很赚钱",
        "我讨厌茶水",
        "你知道体育怎么写吗",
        "你知道犯罪吗",
        "我们需要重视教育的发展",
        "健康有多重要",
        "有哪些娱乐项目",
        "大熊猫属于哺乳动物",

        "动物园会不会养动物",
        "反正犯罪不好",
        "反正我不是知道的",
        "游戏还是不太好",
        "小孩子的教育需要从身边做",
        "你有哪些不良的健康习惯",
        "珠海正在搞体育项目",
        "娱乐也是有讲究的",

        "你有过犯罪吗",
        "我很健康的呢",
        "大人小孩一起娱乐吧",
        "我喜欢养动物",
        "我爱你-我的体育",
        "桌球游戏很不错",
        "小学教育从十岁开始",
        "你怎么能这样呢",

        "你为什么喝茶",
        "从小就应该培养教育习惯",
        "我喜欢赚钱去娱乐呢",
        "网络游戏大不大",
        "给我买一只动物回来",
        "你为什么不去上体育",
        "健康的你和大家",
        "中国的犯罪率是很高的",

        "去了解健康知识吧",
        "国家离不开教育的进步",
        "小孩子玩游戏很不好",
        "我不喜欢美国",
        "国家的体育很厉害",
        "你喜欢花钱娱乐吗",
        "我不想犯罪",
        "你说动物会哭吗",

        "猫科动物比鸟类小很多",
        "游戏可以开发大脑",
        "体育总局在哪里",
        "出去娱乐很花钱",
        "其实一起学习也是教育呢",
        "你好讨厌啊",
        "一起健身一起健康运动",
        "国家正在打击犯罪",

        "中国的体育很给力的",
        "犯罪是什么",
        "我也不知道其它的",
        "你娱乐我我娱乐你",
        "大脑可以玩游戏",
        "你会不会受教育啊",
        "狗狗属于动物呢",
        "吸烟有益健康",

        "大型动物很重的",
        "其实游戏很好",
        "犯罪的影响很大",
        "其它的我也管不了",
        "我就是喜欢体育",
        "生冷食物有益健康成长",
        "QQ娱乐很不错呢",
        "我也去教育别人了呢",

        "让我出去读取回来教育你",
        "你会玩游戏吗",
        "所有动物也喜欢吃草",
        "犯罪特别不好",
        "网球是属于体育项目吧",
        "就是不喜欢你",
        "没有一成不变的健康饮食",
        "就是不想跟你娱乐在一起"
    ]
key_words = ["教育", "健康", "动物", "娱乐", "游戏", "体育", "犯罪"] # 还包含其它
num_key_count = len(key_words)

all_chars = "!教育是国家发展的第一动力在动物园可以看见熊猫吗？为什么要犯罪为什么需要重视健康我要去体育馆我喜欢喝茶我喜欢玩王者游戏你喜欢出去娱乐吗王者游戏很赚钱我讨厌茶水你知道体育怎么写吗你知道犯罪吗我们需要重视教育的发展健康有多重要有哪些娱乐项目大熊猫属于哺乳动物动物园会不会养动物反正犯罪不好反正我不是知道的游戏还是不太好小孩子的教育需要从身边做起你有哪些不良的健康习惯珠海正在搞体育项目娱乐也是有讲究的你有过犯罪吗我很健康的呢大人小孩一起娱乐吧我喜欢养动物我爱你-我的体育桌球游戏很不错小学教育从十岁开始你怎么能这样呢你为什么喝茶从小就应该培养教育习惯我喜欢赚钱去娱乐呢网络游戏大不大给我买一只动物回来你为什么不去上体育健康的你和大家中国的犯罪率是很高的去了解健康知识吧国家离不开教育的进步小孩子玩游戏很不好我不喜欢美国国家的体育很厉害你喜欢花钱娱乐吗我不想犯罪你说动物会哭吗猫科动物比鸟类小很多游戏可以开发大脑体育总局在哪里出去娱乐很花钱其实一起学习也是教育呢你好讨厌啊一起健身一起健康运动国家正在打击犯罪中国的体育很给力的犯罪是什么我也不知道其它的你娱乐我我娱乐你大脑可以玩游戏你会不会受教育啊狗狗属于动物呢吸烟有益健康大型动物很重的其实游戏很好犯罪的影响很大其它的我也管不了我就是喜欢体育生冷食物有益健康成长QQ娱乐很不错呢我也去教育别人了呢让我出去读取回来教育你你会玩游戏吗所有动物也喜欢吃草犯罪特别不好网球是属于体育项目吧就是不喜欢你没有一成不变的健康饮食就是不想跟你娱乐在一起 "  # 字符集
all_char_count = len(all_chars)

print("all_char_count is:", all_char_count)

train_sentences = 5000

char_dim = all_char_count  # 每个字的维度
sentence_length = 20  # 样本文本长度

# 配置参数
epoch_num = 200  # 训练轮数
batch_size = 100  # 每次训练样本个数

train_sample = train_sentences  # 每轮训练总共训练的样本总数
learning_rate = 0.01  # 学习率

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.linear1 = nn.Linear(vector_dim, 32)     #线性层1
        self.activation1 = nn.functional.tanh     #激活层1
        self.linear2 = nn.Linear(32, 64)     #线性层2
        self.activation2 = nn.functional.tanh     #激活层2
        self.linear3 = nn.Linear(64, 8)  # 线性层3
        self.loss = nn.functional.cross_entropy  #loss函数采用均方差损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = self.pool(x.transpose(1, 2)).squeeze() #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)
        x = self.linear1(x)                       #(batch_size, vector_dim) -> (vector_dim, 64)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        y_pred = self.linear3(x)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
def build_vocab():
    chars_set = set(all_chars)
    vocab = {}
    for index, char in enumerate(chars_set):
        vocab[char] = index   #每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

# 根据字符串关键字获取类型ID
# 教育、健康、动物、娱乐、游戏、体育、犯罪和其它
def getLabelIndex(input_string):
    for i in range(num_key_count):
        if input_string.find(key_words[i]) >= 0:
            return i
    # 都没有找到，则默认为其它
    return num_key_count

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(index, vocab, sentence_length):
    #随机生成一个字符串
    x_string = "" #字符串
    nRandLength = np.random.randint(low=5, high=sentence_length - 1) #字符串长度
    nRandKeyWordPos = np.random.randint(low=0, high=nRandLength - 2) #关键词所在位置
    nRandWordIndex = index % (num_key_count + 1) #关键词序号
    #随机产生带有关键字的句子
    for i in range(nRandKeyWordPos):
        x_string += all_chars[np.random.randint(low=0, high=all_char_count - 1)]
    if nRandWordIndex != num_key_count:
        x_string += key_words[nRandWordIndex]
    nStrLength = len(x_string)
    nStrLength = nRandLength - nStrLength
    for i in range(nStrLength):
        x_string += all_chars[np.random.randint(low=0, high=all_char_count - 1)]
    #保证数据长度一致，长度为：sentence_length
    nStrLength = len(x_string)
    nNeedAddCnt = sentence_length - nStrLength
    for n in range(nNeedAddCnt):
        x_string += " "
    #指定哪些字出现时为正样本
    y = [0 for i in range(8)]  # 总共8个分类
    #教育、健康、动物、娱乐、游戏、体育、犯罪和其它
    y[getLabelIndex(x_string)] = 1
    x = [vocab.get(word, vocab['unk']) for word in x_string]   #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(i, vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(train_sentences, vocab, sample_length)   #建立200个用于测试的样本
    #print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), train_sentences - sum(y)))
    #print("本次预测集x:", x, "y:", y)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            # print("训练预测值：", y_p, ", 真实值：", y_t)
            if np.argmax(y_p) != np.argmax(y_t):
                wrong += 1
            else:
                correct += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings, origin_test_strings):
    if len(origin_test_strings) != len(input_strings):
        print("输入的数据长度不一致，终止预测")
        raise ValueError
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测

    nTotalCount = 0
    nRightCount = 0

    for i, input_string in enumerate(input_strings):
        #获取真实类型
        nTotalCount += 1
        nRealType = getLabelIndex(origin_test_strings[i])
        print("测试预测值：", result[i], ", 真实值：", nRealType)

        if np.argmax(result[i]) == nRealType:
            nRightCount += 1

        print("*****输入：%s, 预测类别：%d，真实类别：%d*****" % (origin_test_strings[i], np.argmax(result[i]), nRealType)) #打印结果
    print("*****测试汇总数据:测试样本={},正确样本={},错误样本={},正确率={}.*****".format(nTotalCount, nRightCount, nTotalCount - nRightCount, nRightCount / nTotalCount))


if __name__ == "__main__":
    ans = input("请输入需要进行的选择(0:训练模型,1:预测模型):\n")
    ans = int(ans)
    if 0 == ans:
        main()
    else:
        test_strings = nlp_test_string[:]
        # 保证数据长度一致，长度为：sentence_length
        nLength = len(test_strings)
        for i in range(nLength):
            nNeedAddCnt = sentence_length - len(test_strings[i])
            for n in range(nNeedAddCnt):
                test_strings[i] += " "

        predict("model.pth", "vocab.json", test_strings, nlp_test_string)
