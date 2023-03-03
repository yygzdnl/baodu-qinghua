import json
import random


# 中文 4e00 9fbf 19968-20767 800
# 俄语 0400 052F 1024-1323 300
# 日语 3040 30ff 12352 12451 100
def language_random_index():
    j = 0
    vocab = {}
    rand_index = random.sample(range(0, 21424), 21424)
    for i in range(0x4e00, 0x9fc0):
        char = chr(i)
        vocab[char] = rand_index[j]
        j += 1
    for i in range(0x0400, 0x0530):
        char = chr(i)
        vocab[char] = rand_index[j]
        j += 1
    for i in range(0x3040, 0x3100):
        char = chr(i)
        vocab[char] = rand_index[j]
        j += 1

    writer = open('vocab.json', 'w', encoding='utf8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 生成乱序三国语言字符集
# 削弱index顺序和大小对于语言类别的影响
# 执行一次
language_random_index()
