import jieba
import numpy

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
    n = len( W )

    #建立词典
    Dic = []
    while W :
        word = W[ 0 ]
        Dic.append( word )
        num = W.count( word )
        for i in range( num ) :
            #删除所有第一个元素
            W.remove( word )

    return  Dic

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
    #V是向量空间，行是不同的单词，列是不同的维度
    #W是文本，不同行是不同的句子
    #n是窗口大小
    m = len( V.T )
    hiden_num = 20

    #初始化参数
    H = numpy.random.rand( 100 , n * m )/ 10#输入层到输出层的权重
    U = numpy.random.rand( len(V) , 100 )/ 10
    d = numpy.random.rand( 100 , 1 )/ 10 - 0.05
    b = numpy.random.rand( len(V) , 1 )/ 10 - 0.05

    for i in W :
        for j in range( 0 , len( i ) - ( n + 1 ) ) :
            #生成x
            x = []
            for p in range( j , j + n ) :
                for q in range( 0 , m ) :
                    x.append( V[ i[ p ] , q ] )
            x = numpy.mat( x ).T

            # 前向传播
            o = d + numpy.dot(H, x)
            a = softmax( o )
            y = b + numpy.dot(U , a)

            #反向传播
            delta_b = - numpy.ones( [len( b ) , 1] ) / y.sum()
            delta_b[ j + n ] += 1
            delta_U = numpy.ones( [len(V) , 100] )
            for p in range( 0 , m ) :
                delta_U[ p , : ] = -a.T / y.sum()
            delta_U[ j + n , : ] = delta_U[ j + n , : ] + a.T
            delta_o = ( 1 - numpy.power( a , 2 ) )
            delta_d = numpy.multiply( delta_o , ( numpy.dot( delta_b.T , U ).T ) )
            delta_H = numpy.ones( [100 , n * m] )
            for p in range( 0 , len(H.T) ):
                H[ p , : ] = delta_o[ p ] * x.T
            delta_x = numpy.dot( H.T , delta_d )

            print("show x")
            print(x)
            # print("show a")
            # print(a)

            #更新变量
            b += 0.05 * delta_b
            d += 0.05 *delta_d
            U += 0.05 *delta_U
            H += 0.05 *delta_H
            x += 0.05 *delta_x
            #将更新后的变量x中的数值放回V中更新
            for p in range( 0 , n ) :
                V[ i[p] , : ] = x[ p * n : ( p + 1 ) *n , 0].T
    return V


def softmax( x ) :
    #这里输入的是列向量
    n = len( x )
    y1 = numpy.exp( x ) - numpy.exp( -x )
    y2 = numpy.exp( x ) + numpy.exp( -x )
    y = []
    for i in range( n ) :
        y.append( y1[ i , 0 ] / y2[ i , 0 ] )

    return numpy.mat( y ).T


if __name__ == "__main__" :
    print("*==================================开始==========================================*")
    L = stop_set( "E:\作业\data\stopwords-master\cn_stopwords.txt" )#提取停词表
    path = "E:/作业/data"
    file_name = path + "/data.txt"
    W = data_set( path , L )#去停词化和分词化
    Dic = Dic_set( W )#建立词典

    W = change( W , Dic )#将W转换为Dic中对应的序号表示
    # print( W )

    #生成词向量矩阵
    n = len( Dic )
    m = 5 #给定向量维度为5
    V = numpy.random.rand( n , m ) / 10 - 0.05

    window = 5 #设置窗口数为5
    # print("==============rawV==================")
    # print(V)
    V = NNLM( V , W , window) #通过神经网络算法对向量V进行训练
    print("==============V==================")
    # print(V)


