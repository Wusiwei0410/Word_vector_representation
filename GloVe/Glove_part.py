#这里是在小规模数据上测试的程序

import numpy as np
import math

def f_x( X , x_max , alpha ) :
    # 这里输入的是矩阵X
    Y = np.mat(np.ones(X.shape))
    Y[ np.where( X < x_max ) ] = np.power(Y[ np.where( X < x_max ) ] / x_max , alpha )
    return Y


if __name__ == "__main__" :
    n = 100 #词典长度
    m = 5 #词向量长度
    X = np.mat( np.random.rand( n , n ) ) #随机生成共现矩阵
    W = np.mat( np.random.rand( n , m ) ) #词向量矩阵
    W_yiba = np.mat( np.random.rand( n , m ) ) #上下文矩阵
    b = np.mat( np.random.rand( n , 1 ) ) #偏置项
    b_yiba = np.mat(np.random.rand(n, 1))  # 偏置项
    alpha = 3 / 4
    x_max = 100
    yita = 0.1#步长

    # delta_W =np.mat( np.zeros( [n , m] ) )
    # delta_W_yita =np.mat( np.zeros( [n , m] ) )
    # delta_b = np.mat(np.zeros( n , 1 ))
    # delta_b_yita = np.mat(np.zeros(n, 1))

    # print(np.multiply(f_x(X, x_max, alpha), W) )
    J_ba = np.multiply(f_x(X, x_max, alpha), W* W_yiba.T + (b + b_yiba.T) - np.log(X))  # 计算f*(ww+b+b-log)
    W -= yita * J_ba * W_yiba
    W_yiba -= yita * J_ba.T * W

    # #开始迭代
    # #J_ba = np.multiply(f( X , x_max , alpha ) , W) * W_yiba.T + ( b + b_yita.T ) - np.log( X )#计算f*(ww+b+b-log)
    # f_X = f( X , x_max , alpha )
    # for i in range( n ) :
    #     for j in range( n ) :
    #         tempt = yita * 2 * f_X[i , j] * ( W[ i , : ] * W_yiba[ j , : ].T + b[ i ] + b[ j ] - math.log( X[ i , j ] ) )
    #         W[ i , : ] -= tempt * W_yiba[ j , : ]
    #         W_yiba[j,:] -= tempt * W[i, :]
    #         b[ i ] -= tempt
    #         b_yiba[ j ] -= tempt

    print( W )


