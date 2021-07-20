import numpy as np

if __name__ == "__main__" :
    Doc = [ [ 0 , 1 , 2 , 3 ] , [ 2 , 4 , 1  ,5 ] , [ 0 , 3 , 5 ] ] #也就是说一共有6个单词
    Dic = [ 0 , 1 , 2 , 3 , 4 , 5 ]
    m = 5 #这个是词向量的维度
    n = len(Dic)  #词典中单词个数
    alpha = 1 #搜索步长
    windows = 2 #窗口个数，这里的windows=n-1
    # R = np.mat(np.random.rand(n, m))  # 词向量
    # beta = np.mat(np.random.rand(m, 1))
    # yita = np.mat(np.random.rand(n, 1))
    R = np.mat(np.ones([n, m])) * 0.1 - 0.05
    # for i in range( n ) :
    #     R[ i , : ] *= ( i + 1 )
    # R = np.mat( np.random.rand( n , m ) ) / 10 - 0.05 #词向量
    beta = np.mat(np.ones([m, 1])) / 10 - 0.05
    # beta = np.mat( np.random.rand( m , 1 ) )/ 10 - 0.05
    yita = np.mat(np.ones([n, 1])) / 10 - 0.05
    # yita = np.mat( np.random.rand( n , 1 ) )/ 10 - 0.05

    # C = np.mat( np.random.rand( m , windows * m ) ) - 0.5#生成C
    C = [[5.58644921e-02, 1.49952615e-01, 6.57716706e-02, -1.61199110e-01, -1.98648284e-01, -2.35626251e-01,
          4.38256485e-02, -4.98611234e-01, -1.20204906e-01, -2.80379701e-01],
         [-1.89013233e-01, -3.96008301e-01, -1.73638660e-01, 3.78135764e-01, -6.50261582e-02, 1.39306574e-01,
          -2.76470025e-01, 3.63716450e-01, -1.92569026e-01, 1.97733933e-04],
         [-4.25682991e-01, -6.30517942e-02, 2.19189889e-01, 2.99293867e-01, -2.63840742e-01, 4.02631818e-01,
          -3.20275798e-01, 3.30910998e-01, 2.76544806e-01, 3.67781093e-02],
         [-2.61251017e-01, 2.38209961e-01, -1.04484953e-01, -6.75455763e-02, -1.68792027e-01, 4.69605994e-01,
          1.75535928e-01, 3.09273860e-01, 2.30476259e-01, 4.15225661e-01],
         [2.25574430e-01, -2.22962211e-02, -2.75801102e-01, -1.05909782e-01, 2.97899043e-01, 4.63499185e-01,
          7.76556100e-02, 3.89431624e-01, -2.77696634e-01, -2.96714233e-01]]
    C = np.mat(C)
    x = np.mat( np.zeros( [ windows * m , 1 ] ) ) #初始化x

    print(R)

    for paper in Doc :
        if len(paper) > windows :
            #如果当前文章的长度大于窗口数
            for i in range( len(paper) - windows ) :
                #初始化梯度
                delta_R = np.mat(np.zeros([n, m]))
                delta_beta = np.mat(np.zeros([m, 1]))
                delta_C = np.mat(np.zeros([ m , windows * m]))
                delta_x = np.mat(np.zeros([windows * m, 1]))
                #生成当前窗口下输入x
                for j in range( windows ):
                    x[ j * m : ( j + 1 ) * m ] = R[paper[i + j], :].T

                #前向传播
                h = C * x + beta
                y = R * h + yita
                p = np.exp( y ) / np.sum( np.exp( y ) )

                #反向传播
                delta_y = -p
                delta_y[ paper[i + windows ] ] += 1
                delta_yita = delta_y.copy()

                for j in range( n ) :
                    for t in range( m ) :
                        delta_R[ j , t ] = delta_y[ j ] * h[ t ]
                        delta_beta[ t ] += delta_y[ j ] * R[ j , t ]
                        for k in range(m * windows):
                            delta_C[ t , k ] += delta_y[ j ] * R[ j , t ] * x[ k ]
                            delta_x[ k ] += delta_y[ j ] * R[ j , t ] * C[ t , k ]

                for j in range( windows ) :
                    for t in range( m ) :
                        delta_R[paper[i + j], t ] += delta_x[ j * m + t ]

                #更新
                R += delta_R * alpha
                yita += delta_yita *alpha
                beta += delta_beta * alpha
                C += delta_C * alpha
                print("输出p:")
                print(delta_C)

    print( "输出R:" )
    print( R )