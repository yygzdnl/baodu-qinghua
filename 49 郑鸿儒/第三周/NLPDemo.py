# 文本分类 区分中俄日三国语言
import json
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt


def to_one_hot(target, shape):
    if target is None:
        pass
    else:
        one_hot_target = np.zeros(shape)
        target = np.array(target.cpu())
        for i, t in enumerate(target):
            one_hot_target[i][int(t[0])] = 1
        return one_hot_target


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_len, vocab_len):
        super(TorchModel, self).__init__()
        self.Embedding = nn.Embedding(vocab_len, vector_dim)
        self.pool = nn.AvgPool1d(sentence_len)
        self.classify = nn.Linear(vector_dim, 3)
        self.activation = torch.softmax
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.Embedding(x)
        x = self.pool(x.transpose(1, 2)).squeeze()
        x = self.classify(x)
        y_pred = self.activation(x, dim=0)

        if y is not None:
            y_t = torch.tensor(to_one_hot(y, y_pred.shape), dtype=torch.float)
            y_t = y_t.cuda()
            return self.loss(y_pred, y_t)
        else:
            return [torch.argmax(y_item) for y_item in y_pred]


# 0 中文
# 1 俄语
# 2 日语
# 随机产生某一国语言的字符串
def random_sentence(language, length):
    res = ''
    for i in range(length):
        if 0 == language:
            val = random.randint(0x4e00, 0x9fbf)
        elif 1 == language:
            val = random.randint(0x0400, 0x052f)
        else:
            val = random.randint(0x3040, 0x30ff)
        res += chr(val)
    return res


# 获取字符集
def get_vocab():
    with open('vocab.json', 'br') as f:
        vocab = json.load(f)
    return vocab


# 从自建字符集中获取index
def get_character_from_custom_vocab(sentence_len):
    language = random.randint(0, 2)
    sentence = random_sentence(language, sentence_len)
    vocab = get_vocab()
    return [vocab[char] for char in sentence], language


def build_datatset(sample_len, sentence_len):
    dataset_x = []
    dataset_y = []

    for i in range(sample_len):
        x, y = get_character_from_custom_vocab(sentence_len)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


def evaluate(model, test_sample, sentence_len):
    model.eval()
    x, y = build_datatset(test_sample, sentence_len)
    x, y = x.cuda(), y.cuda()
    Chinese = Russian = Japanese = 0
    correct = wrong = 0
    for i in y:
        if i == 0:
            Chinese += 1
        elif i == 1:
            Russian += 1
        else:
            Japanese += 1
    print("本次测试中文%d条, 俄语%d条, 日语%d条" % (Chinese, Russian, Japanese))

    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测%d个，错误%d个, 正确率：%f" % (correct, wrong, correct/test_sample))
    return correct/test_sample


def main():
    epoch_num = 30
    batch_size = 30
    train_sample = 900
    test_sample = 200
    char_dim = 20
    sentence_len = 8
    learning_rate = 0.001
    vocab = get_vocab()

    model = TorchModel(char_dim, sentence_len, len(vocab))
    model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_datatset(batch_size, sentence_len)
            x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("第%d轮平均loss: %f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, test_sample, sentence_len)
        log.append([acc, np.mean(watch_loss)])

    plt.plot(range(len(log)), [l[0] for l in log], label='acc')
    plt.plot(range(len(log)), [l[1] for l in log], label='loss')
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "language_diff.mdl")
    return


def predict(model_path, inputs):
    char_dim = 20
    sentence_len = 8
    vocab = get_vocab()
    model = TorchModel(char_dim, sentence_len, len(vocab))
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    x = []
    for string in inputs:
        x.append([vocab[char] if char in vocab else vocab['unk'] for char in string])
    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x).cuda())
    for i, string in enumerate(inputs):
        language = 'Chinese' if result[i] == 0 else 'Russian' if result[i] == 1 else 'Japanese'
        print("输入：%s， 预测类别：%d, 名称：%s" % (string, int(result[i]), language))


if __name__ == "__main__":
    # main()
    test_data = ["か゛なゞずちごや", "ҋҍЀѥҚФԝй", "俘侶使习乳中侑傚", "҈ԤѯяӍуВӮ", "АьҟӄІԌѸӂ", "乔偙乨俁丆乐亷僶",
                 "偞丹乗乆傯儞健佱", "じっねょふん぀ぜ", "ӘТӿУӢԛҷѽ", "ゔり゠ょぢぉぷさ", "偔仓亼並傞儋乕体", "ѨЦԁӊӝӶԆҶ",
                 "伮价俊倢倻侠傶乄", "ゖ゜ゖみしゑうだ", "ら゗んげべ぀ぁよ", "倀傮亲俩亷俑丛偙", "せでうげわやぉゃ"]
    predict("language_diff.mdl", test_data)
