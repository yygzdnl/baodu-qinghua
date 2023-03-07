import torch
import torch.nn as nn
import  jieba
from torch.utils.data import  DataLoader
import  numpy as np


class lstm(nn.Module):
    def __init__(self,vocab,input_dim,hidden_size):
        super(lstm, self).__init__()
        self.embedding=nn.Embedding(len(vocab)+1,input_dim)
        self.lstm=nn.LSTM(input_dim,hidden_size,batch_first=True)
        self.classify=nn.Linear(hidden_size,2)
        self.loss=nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self,x,y=None):
        x=self.embedding(x)
        x,_=self.lstm(x)  # rnn返回两个参数
        y_pred=self.classify(x)
        if y is not None:
            return self.loss(y_pred.view(-1, 2), y.view(-1))
        else:
            return y_pred



# 文本转为数字序列
def sequence_to_label(setence,vocab):
    label=[vocab.get(i,vocab['unk']) for i in setence]
    return label

# jieba分词
def jieba_label(sentence):
    words=jieba.lcut(sentence)
    pointer=0
    label=[0]*len(sentence)
    for word in words:
        length=len(word)
        pointer+=length
        label[pointer-1]=1
    return label
# 加载字表
def build_vocab(vocab_path):
    vocab={}
    with open(vocab_path,encoding='utf-8') as f:
        for index,line in enumerate(f):
            char=line.strip()
            vocab[char]=index
        vocab['unk']=index+1
    return vocab

#数据集
class dataset:
    def __init__(self, corpus_path, vocab, max_length):
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.load()


    def load(self):
        self.data=[]
        with open(self.corpus_path,encoding='utf-8') as f:
            for line in f:
                sequence=sequence_to_label(line,self.vocab)           #词转化为向量
                label=jieba_label(line)
                sequence, label = self.padding(sequence, label)
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label)
                self.data.append([sequence,label])                    # 分别是x和y
                if len(self.data) > 5000:
                    break


    def padding(self,sequence,label): #补齐到同一长度
        sequence=sequence[:self.max_length]
        sequence = sequence[:self.max_length]
        sequence += [0] * (self.max_length - len(sequence))
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))

        return sequence, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]




#建立数据集
def build_dataset(corpus_path, vocab, max_length, batch_size):
    dataset_target = dataset(corpus_path, vocab, max_length)
    data_loader = DataLoader(dataset_target, shuffle=True, batch_size=batch_size)
    return data_loader



# 训练
def main():
    epoches=10
    lr=0.003
    batch_size=20
    input_dim=50
    hidden_size=100
    vocab_path = "chars.txt"  # 字表文件路径
    corpus_path = "corpus.txt"  # 语料文件路径
    max_length=15
    vocab=build_vocab(vocab_path)
    data_loader = build_dataset(corpus_path, vocab, max_length, batch_size)  # 建立数据集
    model = lstm(vocab,input_dim,hidden_size)  # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epoches):
        model.train()
        watch_loss = []
        for x,y in data_loader:
            optim.zero_grad()
            loss=model(x,y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

    torch.save(model.state_dict(), "model.pth")
    return


# predict
def predict(model_path,vocab_path, input_strings):
    input_dim=50
    hidden_size=100
    vocab = build_vocab(vocab_path)
    model = lstm(vocab,input_dim, hidden_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for input_string in input_strings:
        #逐条预测
        x = sequence_to_label(input_string, vocab)
        with torch.no_grad():
            result = model.forward(torch.LongTensor([x]))[0]
            result = torch.argmax(result, dim=-1)  #预测出的01序列
            #在预测为1的地方切分，将切分后文本打印出来
            for index, p in enumerate(result):
                if p == 1:
                    print(input_string[index], end=" ")
                else:
                    print(input_string[index], end="")
            print()


if __name__ == "__main__":
    #main()
    input_strings = ["根据泰勒展开的唯一性",
                     "复变函数的微分定义和实变函数相同",
                     "在单连通区域上处处解析",
                     ]
    predict("model.pth", "chars.txt", input_strings)