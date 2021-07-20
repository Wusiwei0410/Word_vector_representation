import  random
import numpy as np
import FNN神经网络语言模型__NNLM.NNML_name as FNN
import  LSTM.LSTM_main as LS

def CW( W , V , C ) :
    n = len( V ) #词典中单词个数
    m = len( V.T )#向量空间的维度
    windows = 2 * C + 1 #计算窗口数
    hiden_dim = 1000 #隐层数

    #初始化参数
    W0 = np.mat( np.random.rand( hiden_dim , windows * m ) ) / 10 - 0.05
    b0 = np.mat( np.random.rand( hiden_dim ,  1 ) )/10 - 0.05
    W1 = np.mat( np.random.rand( 1 , hiden_dim ) ) / 10 - 0.05
    b1 = np.mat( np.random.rand( 1 , 1 ) )/10 - 0.05

    for statement in W :
        for j in range( len( statement ) - 2 * C ) :
            #提取x
            x = np.mat( '1' )
            x_noise = x#噪音数据
            r = random.randrange( n ) #产生随机整数
            while r == j + C :
                #如果产生的随机整数是当前的窗口中心，则重置
                r = random.randrange(n)  # 产生随机整数
            for p in range( windows ) :
                if p == C :
                    x = np.append(x, V[statement[j + p], :].T, axis=0)  # 拼接到一列
                    x_noise = np.append(x_noise, V[r, :].T, axis=0)  # 拼接到一列
                else :
                    x = np.append(x, V[statement[j + p], :].T, axis=0)  # 拼接到一列
                    x_noise = np.append( x_noise , V[ statement[ j + p ] , : ].T , axis= 0 ) #拼接到一列
            x = x[ 1 : ]#删除第一个用来拼接的单元
            x_noise = x_noise[1:]  # 删除第一个用来拼接的单元

            #前向传播
            h = FNN.tanh( np.dot( W0 , x ) + b0 ) #这里的激活函数使用的是tanh
            print( "h" )
            print( h )
            h_noise = FNN.tanh( np.dot( W0 , x_noise ) + b0 )
            y = W1 * h + b1
            y_noise = W1 * h_noise + b1

            #反向传播
            #对于b1而言，求导之后梯度为0，所以这里就不写了
            delta_W1_x = h.T
            delta_W1_x_noise = h_noise.T
            delta_h = W1.T
            delta_h_noise = W1.T

            delta_b0_x = np.multiply( 1 - np.power( h , 2 ) , delta_h )
            delta_W0_x = LS.LSTM_GD( delta_b0_x , W0 , x )
            delta_b0_x_noise = np.multiply( 1 - np.power( h_noise , 2 ) , delta_h_noise )
            delta_W0_x_noise = LS.LSTM_GD(delta_b0_x_noise, W0, x_noise )

            delta_x = np.dot( delta_b0_x.T , W0 ).T
            delta_x_noise = np.dot( delta_b0_x_noise.T, W0 ).T

            #更新
            #因为是以最小化为目标
            W1 = W1 - delta_W1_x_noise + delta_W1_x
            W0 = W0 - delta_W0_x_noise + delta_W0_x
            b0 = b0 - delta_b0_x_noise + delta_b0_x
            x = x - delta_x_noise + delta_x
            x_C_noise = x - delta_x_noise
            x_C = x + delta_x

            #在向量空间上更新
            for p in range( windows ) :
                if p == C :
                    V[statement[j + p], :] = x_C[p * m: (p + 1) * m].T
                    V[r, :] = x_C_noise[p * m: (p + 1) * m].T
                else :
                    V[statement[j + p], :] = x[p * m: (p + 1) * m].T

    return V

if __name__ == "__main__" :
    print("*==================================开始==========================================*")
    L = FNN.stop_set("E:\作业\data\stopwords-master\cn_stopwords.txt")  # 提取停词表
    path = "E:/作业/data"
    file_name = path + "/data.txt"
    W = FNN.data_set(path, L)  # 去停词化和分词化
    Dic = FNN.Dic_set(W)  # 建立词典

    W = FNN.change(W, Dic)  # 将W转换为Dic中对应的序号表示
    print( W )

    n = len( Dic )
    V = np.mat( np.random.rand( n , 5 ) ) / 10 - 0.05#这里定义的向量空间维度是5
    # print("raw_V")
    # print(V)
    C = 1#确定窗口数，n=2C+1
    V = CW( W , V , C )
    print("V")
    print( V )