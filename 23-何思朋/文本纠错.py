

# 网络训练语言模型
import  torch
import  torch.nn as nn
import  numpy as np
import jieba
import random
import torch.utils.data as data

# 训练词向量

class model(nn.Module):
    def __init__(self,vocab,input_dim,hidden_dim):
        super(model, self).__init__()
        self.embedding=nn.Embedding(len(vocab)+1,input_dim)       # batch_size,window_len,input_dim,
        self.rnn=nn.RNN(input_size=input_dim,
                        hidden_size=hidden_dim,
                        batch_first=True,
                        num_layers=3)
        self.dropout = nn.Dropout(0.1)
        self.linear=nn.Linear(hidden_dim,len(vocab)+1)
        self.loss = nn.functional.cross_entropy

    def forward(self,x,y=None):
        x=self.embedding(x)                 # batch_size,word_length,input_dim
        x,_=self.rnn(x)                     # batch_size,word_length,hidden_dim
        x=x[:,-1,:]                         #batch_size,hiiden_dim
        x=self.dropout(x)
        y_pred=self.linear(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return torch.softmax(y_pred)




# 读取vocab
def read_vocab(path):
    vocab={}
    with open(path,encoding='utf-8') as f:
        for index,line in enumerate(f):
            vocab[line[0]]=index
    return vocab


# 样本
def sample(vocab,window_size,corpus):
       start=random.randint(0,len(corpus)-window_size-1)
       end=start+window_size
       window=corpus[start:end]
       target=corpus[end]
       x=[vocab.get(i,vocab['unk']) for i in window]
       y=vocab.get(target,vocab['unk'])
       return x,y


# 数据
def build_dataset(batch_size,sample_num,vocab,window_size,corpus):
    x_data=[]
    y_data=[]
    for i in range(sample_num):
        x,y=sample(vocab,window_size,corpus)
        x_data.append(x)
        y_data.append(y)
    x=torch.LongTensor(x_data)
    y=torch.LongTensor(y_data)
    dataset=data.TensorDataset(x,y)

    loader=data.DataLoader(dataset,shuffle=True,batch_size=batch_size)

    return loader
#加载语料
def load_corpus(corpus):
    return open(corpus,encoding='utf-8').read()
# 测试
def evaluate(model,vocab,corpus):
    sample_num=250
    batch_size=10
    window_size=8
    model.eval()
    loader=build_dataset(batch_size,sample_num,vocab,window_size,corpus)
    right=0
    wrong=0
    for x,y in loader:
        y_pred=model(x)
        s=torch.ones(batch_size,x.shape[1])
        s*=torch.argmax(y_pred)==torch.argmax(y)
        right=sum(s)
        wrong=len(s)-right
    return right/wrong



# 训练
batch_size=20
lr=1e-3
sample_num=3000
window_size=6
input_dim=128
hidden_dim=30
vocab_path='vocab.txt'

vocab=read_vocab(vocab_path)
vocab['unk']=len(vocab)+1
corpus_path='corpus.txt'
corpus=load_corpus(corpus_path)
model=model(vocab,input_dim,hidden_dim)

optimi=torch.optim.Adam(model.parameters(),lr=lr)
loader=build_dataset(batch_size,sample_num,vocab,window_size,corpus)
epoches=10


for epoch in range(epoches):
    model.train()
    watch_loss=[]
    for x,y in loader:
        loss=model(x,y)
        optimi.zero_grad()
        loss.backward()
        optimi.step()
        watch_loss.append(loss.item())
    print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))








# 计算成句概率














# 文本纠错任务



















