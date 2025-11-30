# week3作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

# 待切分文本
sentence = "经常有意见分歧"

# 目标输出;顺序不重要
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut_front(sentence, Dict):
    """
    从前往后做切分，切分出来的数据是反序的
    :param sentence: 切分的字符串
    :param Dict: 字符串切分的词表
    :return: 切分后的所有字符列表
    """
    num = len(sentence)
    # 初始化空列表
    target = [[] for _ in range(num + 1)]
    # 初始化第一个列表为空，保证循环能正常运行
    target[0] = [[]]
    for i in range(1, num+1):
        for j in range(i):
            word = sentence[j:i]
            if word in Dict:
                for seg in target[j]:
                    target[i].append([word] + seg)

    return target[num]


def all_cut_back(sentence, Dict):
    """
      从后往前做切分，切分出来的数据是正序的
      :param sentence: 切分的字符串
      :param Dict: 字符串切分的词表
      :return: 切分后的所有字符列表
      """
    num = len(sentence)
    # 初始化空列表
    target = [[] for _ in range(num+1)]
    # 初始化最后的列表为空，保证循环能正常运行
    target[num] = [[]]
    for i in range(num - 1, -1, -1):
        for j in range(i + 1, num + 1):
            word = sentence[i:j]
            if word in Dict:
                for seg in target[j]:
                    target[i].append([word] + seg)

    return target[0]


text = all_cut_back(sentence, Dict)
sort_list = sorted(text, key=lambda x:len(x), reverse=False)
for i in sort_list:
    print(i)
