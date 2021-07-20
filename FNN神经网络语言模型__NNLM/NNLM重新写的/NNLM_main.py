import  numpy as np
import os
import jieba

def tanh( x ) :
    return (np.exp(x) - np.exp( -x ))/(np.exp(x) + np.exp( -x ))

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
    f = open(path + "/train/" + "体育.txt" , mode='r', encoding='utf-8')
    tempt += f.read()
    f.close()
    All_doc_str = tempt.split('<text>')

    #读取停词表
    f = open( path + "/stop_words_zh.txt" , mode='r', encoding='utf-8' )
    stop_word = f.read()
    stop_word = stop_word.split( '\n' )

    #去停词化
    S = []
    for text in All_doc_str :
        text = jieba.cut( text )#分词
        text_segment = []

        #去停词化
        for word in text :
            if word not in stop_word :
                if word != ' ' and word != '\u3000' and word != "text" and word != "\n":
                    text_segment.append(word)
        if len(text_segment) >= 1:
            S.append(text_segment)#这一步是去除掉空的list
    #建立词典
    Dic = Dic_set(S)
    print( Dic )
    #将Dic进行保存
    f = open("Dic.txt" , mode= 'w' , encoding= "utf-8")
    for word in Dic :
        f.write(word+"\n")
    f.close()


    #STN[词]
    STN = []
    #将每个text中的单词转换为dic中对应的位置序号
    for text in S :
        tempt = []
        for word in text :
            tempt.append( Dic.index( word ) )
        STN.append( tempt )


    N = len(Dic)  #词典中单词个数
    n = 6#窗口数，这里一个窗口的长度是n+1
    m = 10#词向量维度
    hiden_num = 200  # 隐层个数

    #初始化词向量
    d = np.mat( np.random.rand( hiden_num , 1 ) ) -0.5
    H = np.mat( np.random.rand( hiden_num , n * m ) )-0.5

    V = np.mat( np.random.rand( N , m ) )-0.5
    W = np.mat( np.random.rand( N , n * m ) )-0.5
    H = np.mat( H )/ 10 - 0.05
    d = np.mat( d )/ 10 - 0.05
    U = np.mat( np.random.rand( N , hiden_num ) )-0.5
    b = np.mat( np.random.rand( N , 1 ) )-0.5

    alpha = 0.1
    I = np.mat(np.ones( [ 1 , np.shape(W)[0] ] ))  # 形成一个矩阵规模为W的转置的元素全为1的矩阵

    count = 0
    S_len = len(STN)
    for s in STN :
        print( count , "/" , S_len )
        count += 1
        for i in range(len(s) - n ) :
            #前向传播过程--已经验证
            x = V[ s[ i ] , : ].copy()#初始化x
            for j in range( 1 , n ) :
                x = np.hstack( ( x , V[ s[ i + j ] , : ] ) )
            #将前n-1个单词的向量进行拼接,最后得到的x是行向量
            h = tanh( d + H *x.T )#手动验证正确
            # y = b +  U * h
            y = b + W * x.T + U * h

            #softmax--全连接--以验证正确性
            p_tempt = np.exp( y )
            p = p_tempt / p_tempt.sum()

            #反向传播
            onehot_wt = np.mat( np.zeros( [ N , 1 ] ) )
            onehot_wt[ s[ i + n ] ] = 1
            delta_b = onehot_wt - p
            delta_U = delta_b * h.T
            delta_W = delta_b * x
            delta_d = np.multiply( U.T * delta_b , 1 - np.power( h , 2 ) )
            delta_H = delta_d * x
            # delta_x = delta_d.T * H
            delta_x =   delta_d.T * H + delta_b.T * W

            #更新
            b += alpha * delta_b
            W += alpha * delta_W
            U += alpha * delta_U
            d += alpha * delta_d
            H += alpha * delta_H
            x += alpha * delta_x
            # print( delta_x )
            for j in range( n ):
                V[s[i + j], :] = x[0 , range( m * j , m * ( j + 1 ) ) ].copy()
    np.savetxt("V.txt", V, fmt="%f", delimiter=" ")
