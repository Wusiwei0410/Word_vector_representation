import numpy as np
import jieba

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
    path = "C:/Users/90335/Desktop/课题组论文/机器学习课件2020/3.data/Tsinghua"

    tempt = ""
    f = open(path + "/train/" + "体育.txt", mode='r', encoding='utf-8')
    tempt += f.read()
    f.close()
    All_doc_str = tempt.split('<text>')

    # 读取停词表
    f = open(path + "/stop_words_zh.txt", mode='r', encoding='utf-8')
    stop_word = f.read()
    stop_word = stop_word.split('\n')

    # 去停词化
    S = []
    for text in All_doc_str:
        text = jieba.cut(text)  # 分词
        text_segment = []

        # 去停词化
        for word in text:
            if word not in stop_word:
                if word != ' ' and word != '\u3000' and word != "text" and word != "\n":
                    text_segment.append(word)
        if len(text_segment) >= 1:
            S.append(text_segment)  # 这一步是去除掉空的list
    # 建立词典
    Dic = Dic_set(S)
    print(Dic)
    # 将Dic进行保存
    f = open("Dic.txt", mode='w', encoding="utf-8")
    for word in Dic:
        f.write(word + "\n")
    f.close()

    # STN[词]
    STN = []
    # 将每个text中的单词转换为dic中对应的位置序号
    for text in S:
        tempt = []
        for word in text:
            tempt.append(Dic.index(word))
        STN.append(tempt)

    N = len(Dic)  # 词典中单词个数
    windows = 6  # 窗口数，这里一个窗口的长度是n+1
    m = 10  # 词向量维度
    alpha = 0.1  # 搜索步长
    n = len(Dic)




    R = np.mat(np.random.rand(n, m)) - 0.5 # 词向量
    beta = np.mat(np.random.rand(m, 1))- 0.5
    yita = np.mat(np.random.rand(n, 1))- 0.5
    # R = np.mat( np.random.rand( n , m ) ) / 10 - 0.05 #词向量
    # beta = np.mat( np.random.rand( m , 1 ) )/ 10 - 0.05
    # yita = np.mat( np.random.rand( n , 1 ) )/ 10 - 0.05

    print( STN )



    C = []
    for i in range( m ) :
        C.append(np.mat(np.random.rand(windows, m)) - 0.5 )
        # C.append( np.mat( np.random.rand( windows , m ) ) / 10 - 0.05 )

    RP = np.mat( np.zeros( [ windows , m ] ) ) #专门用来保存当前窗口下的词向量
    h = np.mat( np.zeros( [ m , 1 ] ) ) #初始化h，因为python中没有针对三维矩阵的计算，需要迭代

    print(R)

    for paper in STN :
        if len(paper) > windows :
            #如果当前文章的长度大于窗口数
            for i in range( len(paper) - windows ) :
                PI = []  # 记录当前窗口下单词的下标
                for j in range( windows ) :
                    PI.append(paper[i + j])
                    RP[j, :] = R[paper[i + j], :].copy()

                #前向传播
                for l in range( m ) :
                    h[ l ] = np.sum( np.multiply( RP , C[ l ] ) ) + beta[ l ]
                y = R * h  + yita
                # y = R * h / windows + yita
                p = np.exp( y ) / np.sum( np.exp( y ) )

                #反向传播
                delta_y = -p
                delta_y[ paper[i + windows ] ] += 1
                delta_yita = delta_y.copy()
                delta_beta = R.T * delta_y

                delta_C = []
                for l in range( m ) :
                    delta_C.append( ( R[ : , l ].T * delta_y )[0 , 0] * RP )

                delta_R = np.mat( np.zeros( R.shape ) )

                # 如果第n个单词是在前面n-1个单词里出现过的
                for t in range(len(PI)):
                    a = PI[t]
                    CL = np.mat(np.zeros([m, m]))  # 因为python没有三维矩阵，这里需要固定i,把C中数据扣下来
                    for l in range(m):
                        CL[l, :] = C[l][t, :].copy()#将C的所有列表中的第t行全部提取出来
                    delta_R[a, :] = delta_y.T * R * CL


                #计算修正项
                Correct = np.mat( np.zeros([ 1 , m ]) )#初始化修正项
                for b in range(m):
                    Correct[0 , b] = beta[b] + np.sum(np.multiply(RP, C[b]))

                #对delta_R进行修正
                delta_R += delta_y * Correct

                index = np.where( delta_R > 1 )


                #更新
                R += delta_R * alpha
                # R += delta_R * alpha / windows
                yita += delta_yita *alpha
                beta += delta_beta * alpha
                for l in range( m ) :
                    # C[l] += delta_C[l]
                    C[ l ] += alpha * delta_C[ l ]
                print(  "输出一下：" )
                print(R)
                print(yita)
                print(beta)

    print( R )
    print( yita )
    print( beta )