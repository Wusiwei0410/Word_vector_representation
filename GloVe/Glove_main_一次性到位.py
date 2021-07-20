#这里是在小规模数据上测试的程序
import numpy as np
import re
import math

def f_x( X , x_max , alpha ) :
    # 这里输入的是矩阵X
    Y = np.mat(np.ones(X.shape))
    Y[ np.where( X < x_max ) ] = np.power(Y[ np.where( X < x_max ) ] / x_max , alpha )
    return Y

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
    n = len(Dic)  # 词典长度
    print(Dic)
    print( n )

    STN = []
    # 将每个text中的单词转换为dic中对应的位置序号
    for text in Doc:
        tempt = []
        for word in text:
            tempt.append(Dic.index(word))
        STN.append(tempt)

    #计算共现矩阵X
    X = np.mat( np.zeros( [ n , n ] ) )
    count_paper = len(STN)
    count = 0
    for text in STN :
        print( count , '/' , count_paper )
        count += 1
        #提取出这段话中的所有单词
        word_all = []
        while text:
            word = text[0]
            word_all.append(word)
            num = text.count(word)
            for i in range( num ) :
                text.remove( word )
        for word1 in word_all :
            for word2 in word_all :
                if word1 != word2 :
                    X[ word1 , word2 ] += 1
    X = (X + 1)/ (count_paper+1)

    m = 5 #词向量长度
    W = np.mat( np.random.rand( n , m ) ) #词向量矩阵
    W_yiba = np.mat( np.random.rand( n , m ) ) #上下文矩阵
    b = np.mat( np.random.rand( n , 1 ) ) #偏置项
    b_yiba = np.mat(np.random.rand(n, 1))  # 偏置项
    alpha = 3 / 4
    x_max = 100
    yita = 0.1#步长

    #开始迭代
    J_ba = np.multiply(f_x(X, x_max, alpha), W * W_yiba.T + (b + b_yiba.T) - np.log(X))  # 计算f*(ww+b+b-log)
    W -= yita * J_ba * W_yiba
    W_yiba -= yita * J_ba.T * W

    print( W )
    print( W_yiba )


