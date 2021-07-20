import jieba
import numpy
import math

def stop_set( path ) :
    f = open(path, 'r', encoding='utf-8')
    L = []
    for word in f.readlines():
        L.append(word.strip())#默认的方法去除换行符

    f.close()
    return L

#对文本进行去停词化和分词
def data_set(path , L ):
    #L是提取的停词表
    #这里输入的只是数据所在的文件夹名，
    # 统一规定文本文件的名字为data.txt
    file_name = path+"/data.txt"

    #注意，读取中文时记得规定解码方式
    f = open(file_name,'r',encoding='utf-8')

    #按行进读取输出
    line = f.read()

    R = ''
    for i in line :
        R += i

    #进行分词化
    seg_ment = jieba.cut( R )

    W = []
    tempt = []
    # 去停词
    for word in seg_ment:
        if word == '。' :
            #以。作为句子结尾的标记
            #进行以句子为单位的划分
            W.append( tempt )
            tempt = []

        if word not in L :
            #对每一行的每一个字，如果不在停词表中
            tempt.append(word)

    f.close()
    return  W

#建立词典
def Dic_set( W ) :
    #W是去停词化和分词化后的文本
    #首先将所有的句子合并到一起
    tempt = []
    for i in W :
        for j in i :
            tempt.append( j )

    W = tempt

    #建立词典
    Dic = []
    frequence = [] #频率
    while W :
        word = W[ 0 ]
        Dic.append( word )
        num = W.count( word )
        frequence.append( num )
        for i in range( num ) :
            #删除所有第一个元素
            W.remove( word )

    return  Dic , frequence

#文本转化为词典中对应的序号
def change( W , Dic ) :
    n = len( W )
    D = []
    for i in W :
        tempt = []
        for j in i :
            tempt.append( Dic.index( j ) )
        D.append( tempt )
    return D

def NNLM( V , W , n ) :
    alpha = 1 #学习率
    #V是向量空间，行是不同的单词，列是不同的维度_mat
    #W是文本，不同行是不同的句子_list
    #n是窗口大小
    m = len( V.T )
    hiden_num = 100

    #初始化参数
    H = numpy.random.rand( hiden_num , n * m ) - 0.5 #输入层到输出层的权重
    U = numpy.random.rand( len(V) , hiden_num ) - 0.5
    d = numpy.random.rand( hiden_num , 1 ) - 0.5
    b = numpy.random.rand( len(V) , 1 ) - 0.5

    for i in W :
        for j in range( 0 , len( i ) -n ) :
            #生成x
            x = []
            for p in range( j , j + n ) :
                for q in range( 0 , m ) :
                    x.append( V[ i[ p ] , q ] )
            x = numpy.mat( x ).T

            # 前向传播
            o = d + numpy.dot(H, x)
            a = tanh( o )
            y = b + numpy.dot(U , a)
            p = exp( y )

            #反向传播
            delta_b = -1 *  p / p.sum()
            delta_b[ i[ j + n ] ] += 1

            delta_U = 1 * U
            for p in range( len( U ) ) :
                delta_U[ p , : ] = delta_b[ p ] * a.T

            delta_d = ( 1 - numpy.power( a , 2 ) )
            delta_d = numpy.multiply( delta_d , ( numpy.dot( delta_b.T , U ).T ) )
            # delta_d = delta_o
            delta_H = 1 * H
            for p in range( 0 , len(H) ):
                delta_H[ p , : ] = delta_d[ p ] * x.T
            delta_x = numpy.dot( H.T , delta_d )

            #更新变量
            b += delta_b * alpha
            d += delta_d * alpha
            U += delta_U * alpha
            H += delta_H * alpha
            x += delta_x * alpha
            #将更新后的变量x中的数值放回V中更新
            for p in range( 0 , n ) :
                V[ i[p] , : ] = 1 * x[ p * n : ( p + 1 ) *n ].T
    return V

def tanh( x ) :
    #这里输入的是列向量
    n = len( x )
    y1 = exp(x) - exp(-x)
    y2 = exp(x) + exp(-x)
    y = []
    for i in range(n):
        y.append(y1[i, 0] / y2[i, 0])

    return numpy.mat( y ).T

def exp( x ) :
    y = 1 * x#赋值给y，这样后面的操作不会因为地址相同影响到x
    #注意，这边传入的形参是地址或者说是指针
    for i in range( len( x ) ) :
        if x[i] > 100 :
            y[i] = math.exp(100)
        else :
            y[i] = math.exp(x[ i ])
    return  y

def Delete( Dic , W , fre , t ) :
    n = len( Dic ) #字典长度
    T = []
    tol = 0.995
    for i in range( n ) :
        if ( 1 - ( t / fre[i] ) ** 0.5 ) < tol:
            T.append( Dic[ i ] )
        else :
            #在W中删除该词
            word = Dic[ i ]
            for p in range( len( W ) ) :
                count = W[p].count(word)
                for q in range( count ) :
                    W[ p ].remove(word)
    return T


if __name__ == "__main__" :
    print("*==================================开始==========================================*")
    L = stop_set( "E:\作业\data\stopwords-master\cn_stopwords.txt" )#提取停词表
    path = "E:/作业/data"
    file_name = path + "/data.txt"
    W = data_set( path , L )#去停词化和分词化
    print( W )
    Dic , frequence  = Dic_set( W )#建立词典
    t = 1e-4
    # Dic = Delete( Dic , frequence , t )
    W = change( W , Dic )#将W转换为Dic中对应的序号表示
    # print( W )

    #生成词向量矩阵
    n = len( Dic )
    print( n )
    m = 5 #给定向量维度为5
    V = numpy.random.rand( n , m ) - 0.5

    window = 5 #设置窗口数为5
    print("==============rawV==================")
    print(V)
    V = NNLM( V , W , window) #通过神经网络算法对向量V进行训练
    print("==============V==================")
    print(V)

    print( Dic )
    print( len( Dic) )


