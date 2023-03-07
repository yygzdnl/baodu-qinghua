
# 读取文件，生成词典
# 字典树(前缀)
# 通过语料直接形成字典树
def load_prefix_word_dict(path):
    prefix_word_dict={}
    with open(path,encoding='utf-8') as f:
        for line in f:
            word = line.split()[0]
            for i in range(1,len(word)):
                if word[:i] not in prefix_word_dict:
                    prefix_word_dict[word[:i]]=0  #表示不是词，只是前缀
                prefix_word_dict[word]=1    # 表示是词
    return prefix_word_dict




# 依照前缀词典生成有向无环图

def dag(text,word_dict):
    '''
    :param text:语料
    :param word_dict:前缀词典
    :return:有向无环图
    '''
    dag_dict=dict()
    length=len(text)
    for i in range(length):
        hou_list=list()
        word=text[i]
        k=i
        while k<length and word in word_dict.keys():
            hou_list.append(k)
            k+=1
            word=text[i:k+1]
        if not hou_list:
            hou_list.append(i)
        dag_dict[i]=hou_list
    return dag_dict






# 依照词典对句子进行完全切分
def print_all(list1,node,ans,dict,last_index,string_target,res):
    '''

    :param list1:记录节点是否访问
    :param node: 节点的index
    :param ans:
    :param dict:有向无环图的字典
    :param last_index: 结束节点的index
    :param string_target: 分词目标
    :return:
    '''
    list1[node]=1
    if node==last_index:
        ans_new=list.copy(ans)
        res.append(ans_new)
        return
    for nodes in dict.get(node+1):
        if list1[nodes]==1:
            continue
        else:
            ans.append(string_target[node+1:nodes+1])
            print_all(list1,nodes,ans,dict,last_index,string_target,res)
            list1[nodes]=0
            ans.pop()



def main():
    pre_dict=load_prefix_word_dict('dic.txt')
    text='我毕业于南方科技大学'
    dag_dict=dag(text,pre_dict)
    list1=[0]*len(text)
    length=len(text)-1
    node=0
    res=[]
    ans=[]
    ans.append(text[0])
    print_all(list1,node,ans,dag_dict,length,text,res)
    [print('/'.join(i)) for i in res]


if __name__ == '__main__':
    main()

