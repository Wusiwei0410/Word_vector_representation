import  numpy as np
import  re

#建立词典
def Dic_set( W ) :
    #W是去停词化和分词化后的文本
    #首先将所有的句子合并到一起
    tempt = []
    for text in W :
        for word in text :
            tempt.append(word)
    #将所有的词放在一起
    T = tempt.copy()

    #建立词典
    Dic = []
    while T :
        #当T中单词被删完时，迭代结束
        word = T[ 0 ]
        Dic.append( word )
        num = T.count( word )
        for i in range( num ) :
            #删除所有第一个元素
            T.remove( word )
    return  Dic

if __name__ == "__main__" :
    # 读取停词表
    f = open("stop_words_zh.txt", mode='r', encoding='utf-8')
    L_stop = f.read()
    f.close()
    L_stop = L_stop.split('\n')

    # 读取文本文件
    tempt = ""
    f = open("体育.txt", mode='r', encoding='utf-8')
    R = f.read()
    f.close()

    for j in R:
        if j != '\n' and j != '\u3000' and j.isdigit() == False:
            tempt += j
        else:
            if j == '\n':
                tempt += ' '
    Doc_text = re.findall('<text>.+?</text>', tempt)  # 记录当前文档下的所有文章
    Doc = []
    for j in Doc_text:
        # 对每个文章进行分词化和去停词
        # 进行分词化
        str_pattern = re.compile("【.+.】")
        seg_ment = str_pattern.sub('', j)
        seg_ment = seg_ment.split(' ')
        # 去停用词
        tempt = []
        for word in seg_ment:
            if word not in L_stop and word != '<text>' and word != '</text>' and word != '':
                tempt.append(word)
        Doc.append(tempt)
    print(Doc)

    # 建立词典
    Dic = Dic_set(Doc)
    print(Dic)

    STN = []
    # 将每个text中的单词转换为dic中对应的位置序号
    for text in Doc:
        tempt = []
        for word in text:
            tempt.append(Dic.index(word))
        STN.append(tempt)

    n = len(Dic)  # 这个是词典中词的个数
    m = 5  # 词向量的长度
    hiden_num = 5  # 隐藏层的个数
    windows = 5  # windows = 2 n + 1——这里的n不是字典个数，窗口中心词两边的词的个数
    C = int( windows / 2 )
    alpha = 0.1

    # 初始化
    V = np.mat(np.random.rand(n, m)) - 0.5

    num = len(STN)
    count = 0

    for sentence in STN:
        if len(sentence) >= windows:
            # 如果句子长度不小于窗口长度
            for i in range(len(sentence) - windows + 1):
                L = []
                for j in range(windows):
                    L.append(sentence[i + j])  # 记录当前窗口下所有词的下标
                x = (sum(V[L]) - V[L[C]]) / (windows - 1)  # 计算平均值_行向量

                # 前向传播
                y = V * x.T
                p = np.exp(y) / np.exp(y).sum()

                # 反向传播
                delta_y = -p
                delta_y[L[C]] += 1

                delta_V = delta_y * x
                delta_x = delta_y.T * V

                # 更新
                V += alpha * delta_V
                for j in range(windows):
                    if j != C:
                        V[L[j], :] += alpha * delta_x / (windows - 1)
    print( V )