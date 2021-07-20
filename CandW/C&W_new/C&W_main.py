#这个是直接在数据集上跑的主函数
import numpy as np
import re

def HardThah( x ) :
    #这里输入的是向量
    y = x.copy()
    y[ np.where( x < -1 ) ] = -1
    x[ np.where( x > 1) ] = 1
    return y

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
    f = open( "stop_words_zh.txt", mode='r', encoding='utf-8')
    L_stop = f.read()
    f.close()
    L_stop = L_stop.split('\n')

    #读取文本文件
    tempt = ""
    f = open( "法律.txt", mode='r', encoding='utf-8' )
    R = f.read()
    f.close()

    for j in R :
        if j != '\n' and j != '\u3000' and j.isdigit() == False :
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

    n = len(Dic)  #这个是词典中词的个数
    m = 5#词向量的长度
    hiden_num =5 #隐藏层的个数
    windows = 5 #windows = 2 n + 1——这里的n不是字典个数，窗口中心词两边的词的个数
    alpha = 0.1

    #初始化
    V = np.mat( np.random.rand( n , m ) ) - 0.5
    W = np.mat( np.random.rand( hiden_num , m * windows ) )- 0.5
    U = np.mat( np.random.rand( hiden_num , 1 ) )- 0.5
    b = np.mat( np.random.rand( hiden_num , 1 ) )- 0.5

    for sentence in STN:
        if len(sentence) >= windows:
            # 只有在句子长度不小于窗口长度才进行传播
            for i in range(len(sentence) - windows + 1):
                # 将当前窗口内单词提取出来，拼接成x
                x = V[sentence[i], :].copy()
                for j in range(1, windows):
                    x = np.hstack((x, V[sentence[i + j], :]))  # 最后得到的是行向量

                # 生成随机样本进行对抗
                rand_sample = np.random.randint(0, n)  # 从0~n中生成随机整数
                while rand_sample == sentence[i + int(windows / 2)]:
                    # 如果随机样本和窗口中心词相同，重新生成
                    rand_sample = np.random.randint(0, n)  # 从0~n中生成随机整数

                # 生成对抗x
                x_rand_sample = x.copy()
                x_rand_sample[0, int(windows / 2) * m: (int(windows / 2) + 1) * m] = V[rand_sample,
                                                                                     :]  # python中的int是向下取整的

                # 开始前向传播
                l = W * x.T + b
                h = HardThah(l)
                f = U.T * h
                l_rand_sample = W * x_rand_sample.T + b
                h_rand_sample = HardThah(W * x_rand_sample.T + b)
                f_rand_sample = U.T * h_rand_sample

                # 反向传播
                if max(0, 1 - f + f_rand_sample) > 0:
                    # 当loss function的值大于0时才反向传播
                    delta_U = h - h_rand_sample
                    one_l = np.mat(np.ones([hiden_num, 1]))
                    one_l[np.where(l < -1)] = 0
                    one_l[np.where(l > 1)] = 0
                    one_l_rand_sample = np.mat(np.ones([hiden_num, 1]))
                    one_l_rand_sample[np.where(l < -1)] = 0
                    one_l_rand_sample[np.where(l > 1)] = 0
                    delta_W = np.multiply(U, one_l) * x - np.multiply(U, one_l_rand_sample) * x_rand_sample
                    delta_b = np.multiply(U, one_l - one_l_rand_sample)
                    delta_x = np.multiply(U, one_l).T * W
                    delta_x_rand_sample = np.multiply(U, one_l_rand_sample).T * W

                    # 更新
                    U += alpha * delta_U
                    W += alpha * delta_W
                    b += alpha * delta_b
                    for j in range(windows):
                        if j != int(windows / 2):
                            # 更新x和x'中相同的部分
                            V[sentence[i + j]] += alpha * (
                                        delta_x[0, j * m: (j + 1) * m] - delta_x_rand_sample[0, j * m: (j + 1) * m])
                        else:
                            V[sentence[i + j]] += alpha * delta_x[0, j * m: (j + 1) * m]
                            V[rand_sample] -= alpha * delta_x_rand_sample[0, j * m: (j + 1) * m]
                    #
                    # V[ rand_sample , : ] -= alpha * delta_x_rand_sample
                    # V[ sentence[ i + int( windows / 2 ) ] , : ] += alpha * delta_x
    print( V )