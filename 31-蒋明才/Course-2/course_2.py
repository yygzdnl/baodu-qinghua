"""
 #
 # @Author: jmc-蒋明才
 # @Date: 2023/2/19 22:37
 # @Version: v1.0
 # @Description: 基于文本的分类 -- 正样本为包含字母Aa的字符串  负样本为不包含Aa的样本
                                 例如：  Adnfh  1    jfSdD  0
"""
from loguru import logger
import random
import torch
import torch.nn as nn
from torch.utils.data import dataset, dataloader
from sklearn.metrics import accuracy_score


# 构建数据集  --- 训练集和测试集
def build_dataset(sample_len: int):
    # a-z  97-122  A-Z  65-90  ASCII码范围
    # 假设每个序列的长度为 10
    ds = []
    for _ in range(sample_len):
        len_1 = random.randint(1, 6)
        len_2 = 10 - len_1
        if random.random() > 0.5:
            seq1 = [chr(cr) for cr in random.sample(range(65, 91), k=len_1)]
            seq2 = [chr(cr) for cr in random.sample(range(97, 123), k=len_2)]
            seq = seq1 + seq2
            random.shuffle(seq)
            r_idx = random.randint(0, 9)
            seq[r_idx] = "a"
        else:
            seq1 = [chr(cr) for cr in random.sample(range(66, 91), k=len_1)]
            seq2 = [chr(cr) for cr in random.sample(range(98, 123), k=len_2)]
            seq = seq1 + seq2
            random.shuffle(seq)
        seq = "".join(seq)
        if 'a' in seq.lower():
            ds.append([seq, 1])
        else:
            ds.append([seq, 0])
    return ds


# 自定义加载数据集
class CustomDataset(dataset.Dataset):
    def __init__(self, ds: list):
        super(CustomDataset, self).__init__()
        self.ds = ds

    def __getitem__(self, item):
        x, y = self.ds[item]
        x = x.lower()
        x = [(ord(ele) - 97) for ele in x]
        # Type: list[int] int
        return torch.LongTensor(x), y

    def __len__(self):
        return len(self.ds)


# 构建模型
class BuildModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers=1, bidirectional=False, pooling="mean"):
        super(BuildModel, self).__init__()
        self.pooling = pooling
        self.embedding = nn.Embedding(num_embeddings=27, embedding_dim=input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        if bidirectional:
            self.classifier = nn.Linear(2*hidden_dim, out_dim)
        else:
            self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch_sample):
        # batch_sample: (batch, seq_len)
        batch_embedding = self.embedding.forward(batch_sample)  # (batch, seq_len, input_dim)
        out, _ = self.lstm.forward(batch_embedding)  # (batch, seq_len, hidden_dim)
        if self.pooling == "mean":
            out = torch.avg_pool1d(out.transpose(1, 2), kernel_size=out.size(1)).squeeze(-1)  # (batch, hidden_dim)
        elif self.pooling == "max":
            out = torch.max_pool1d(out.transpose(1, 2), kernel_size=out.size(1)).squeeze(-1)
        else:
            raise ValueError("Pooling的取值为 mean | max")
        out = self.classifier.forward(out)  # (batch, 2)
        # print(out.shape)
        return out


# 模型验证
def dev(model, dev_loader):
    model.eval()
    with torch.no_grad():
        pred = []
        target = []
        for idx, (x, y) in enumerate(dev_loader):
            out = model.forward(x)
            out = torch.softmax(out, dim=1)
            out = torch.argmax(out, dim=1)
            pred += out.tolist()
            target += y.tolist()
        acc = accuracy_score(target, pred)
        return acc


# 训练
def train():
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = 128
    hidden_dim = 64
    out_dim = 2
    bidirectional = True
    num_layers = 1
    pooling = "mean"
    lr = 2e-3
    save_step = 20
    epoch = 10
    save_path = "./checkpoint/class.pt"

    train_ds = build_dataset(1600)
    train_loader = dataloader.DataLoader(CustomDataset(train_ds), shuffle=True, batch_size=batch_size)

    test_ds = build_dataset(400)
    test_loader = dataloader.DataLoader(CustomDataset(test_ds), shuffle=False, batch_size=batch_size)

    model = BuildModel(input_dim,
                       hidden_dim,
                       out_dim,
                       pooling=pooling,
                       bidirectional=bidirectional,
                       num_layers=num_layers).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    best_acc = 0
    for e in range(1, epoch+1):
        for idx, (x, y) in enumerate(train_loader):
            out = model.forward(x.to(device))
            loss = loss_func.forward(out, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            if (idx + 1) % save_step == 0:
                dev_acc = dev(model, test_loader)
                logger.info(f"Epoch: {e}/{epoch}, Step: {idx}/{len(train_loader)}, Loss: {loss},"
                            f" Dev_acc: {dev_acc}")
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    torch.save(model.state_dict(), save_path)
                    logger.info("save model")
            optimizer.step()


# 测试
def model_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BuildModel(input_dim=128,
                       hidden_dim=64,
                       out_dim=2,
                       pooling="mean",
                       bidirectional=True,
                       num_layers=1).to(device)
    model.load_state_dict(torch.load("./checkpoint/class.pt"))
    test = "df好rdfKO你d好k"
    x = test.lower()
    ipts = []
    for ele in x:
        encode = ord(ele)
        if 97 <= encode <= 122:
            ipts.append(ord(ele) - 97)
        else:
            ipts.append(26)
    print(model.forward(torch.LongTensor([ipts]).to(device)))


if __name__ == '__main__':
    train()
    model_test()