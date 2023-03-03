import json
import random

import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
	#embedding_dim 每个字符串向量后的向量维度 ，vocab  #通常对于nlp任务，此参数为字符集字符总数
	def __init__(self,embedding_dim,sentence_length,vocab):
		super(TorchModel, self).__init__()
		self.embedding=nn.Embedding(len(vocab),embedding_dim)#embedding层
		self.pooling=nn.AvgPool1d(sentence_length)#sentence_length需要池化的维数
		self.layer=nn.Linear(embedding_dim,1)
		self.activation=torch.sigmoid
		self.loss=nn.functional.mse_loss

	def forward(self,x,y=None):
		x=self.embedding(x)
		#因为池化层默认将输入张量的最后一维进行池化，因为本例子需要将四维转换为一维，而不是五，所以需要进行转换
		# x=x.transpose(1,2)
		# x=x.squeeze()#去掉最后一维
		# x=self.pooling(x)#进行池化
		x = self.pooling(x.transpose(1, 2)).squeeze()
		x=self.layer(x)
		y_pred=self.activation(x)
		if y is not None:
			return self.loss(y_pred,y)
		else:
			return y_pred

#字符串随便挑了一些字符，实际上还可以扩充
#为每个字生成一个标号
#("a":1,“"b:2")
def build_vocab():
	chars="abcdefghijklmnopqrstuvwxyz"
	vocab={}
	for index,char in enumerate(chars):
		vocab[char]=index
	vocab['unk']=len(vocab)
	return vocab



#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    if set("abc") & set( x) and "x" not in x and  "y" not in x and "z" not in x :
        y = 1
    #指定字都未出现，则为负样本
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y


def has_duplicate_chars(lst):
    for i in range(len(lst) - 1):
        if lst[i] == lst[i + 1]:
            return True
    return False

#j建立数据集
#输入需要的样本数量，需要多少生成多少
def build_dataset(sample_length,vocab,sentence_length):
	'''

	:param sample_length:
	:param vocab:
	:param sentence_length:
	:return:
	'''
	dataset_x=[]
	dataset_y=[]
	for i in range(sample_length):
		x,y=build_sample(vocab,sentence_length)
		dataset_x.append(x)
		dataset_y.append([y])
	return torch.LongTensor(dataset_x),torch.FloatTensor(dataset_y)


#建立模型
def build_model(vocab,char_dim,sentence_length):

	model=TorchModel(char_dim,sentence_length,vocab)
	return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model,vocab,sentence_length):
	model.eval()
	x,y=build_dataset(200,vocab,sentence_length)#建立200个用于测试的样本
	print("本次预测集中共有%d个正样本,%d个负样本"%(sum(y),200-sum(y)))
	correct,wrong=0,0
	with torch.no_grad():
		y_pred=model(x)#模型预测的值
		for y_p,y_t in zip(y_pred,y):#进行与真是标签进行对比
			if float(y_p)<0.5 and int(y_t)==0:
				correct+=1 #负样本
			elif float(y_p)>=0.5 and int(y_t)==1:
				correct+=1
			else:
				wrong+=1

	print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
	return correct / (correct + wrong)

def main():
	#配置参数
	epoch_num=20#训练轮数
	batch_size=20#每次训练样本个数
	train_sample=1000#每轮训练总共训练的样本总数
	char_dim=20#每个字的维度
	sentence_length=6#样本文本长度
	learning_rate=0.005#学习率
	#建立字表
	vocab=build_vocab()
	#建立模型
	model=build_model(vocab,char_dim,sentence_length)
	#选择优化器
	optim=torch.optim.Adam(model.parameters(),lr=learning_rate)
	#记录loss
	log=[]
	#训练过程
	for epoch in range(epoch_num):
		model.train()
		watch_loss=[]
		for batch in range (int(train_sample/batch_size)):
			x,y=build_dataset(batch_size,vocab,sentence_length)
			optim.zero_grad()#梯度归零
			loss=model(x,y)#计算loss
			loss.backward()#计算 梯度
			optim.step()#g更新权重
			watch_loss.append(loss.item())#添加权重
		print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
		acc=evaluate(model,vocab,sentence_length)
		log.append([acc,np.mean(watch_loss)])

	plt.plot(range(len(log)),[l[0] for l in log],label="acc")
	plt.plot(range(len(log)), [l[1] for l in log], label="loss")

	plt.legend()
	plt.show()
	#保存模型
	torch.save(model.state_dict(),"model.pth")
	#保存此表
	writer=open("vocab.json","w",encoding="utf8")
	writer.write(json.dumps(vocab,ensure_ascii=False,indent=2))
	writer.close()
	return

#预测保存的模型
def predict(model_path,vocab_path,input_strings):
	char_dim = 20  # 每个字的维度
	sentence_length = 6  # 样本文本长度
	vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
	model = build_model(vocab, char_dim, sentence_length)  # 建立模型
	model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
	x = []
	for input_string in input_strings:
		x.append([vocab[char.lower()] for char in input_string])  # 将输入序列化
	model.eval()  # 测试模式

	with torch.no_grad():  # 不计算梯度
		result = model.forward(torch.LongTensor(x))  # 模型预测
	for i, input_string in enumerate(input_strings):
		print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i]))  # 打印结果


if __name__ == "__main__":
   	 #main()
	test_strings = ["sfname", "wwzbfg", "rqycbg", "naawww"]
	predict("model.pth", "vocab.json", test_strings)
