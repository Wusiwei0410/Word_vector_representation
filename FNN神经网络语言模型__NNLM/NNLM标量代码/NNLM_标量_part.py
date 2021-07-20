import  numpy as np

def tanh( x ) :
    return (np.exp(x) - np.exp( -x ))/(np.exp(x) + np.exp( -x ))


if __name__ == "__main__" :
    #这里先使用少量数据进行测试
    STN = [ [ 0 , 1 , 2 , 3 ] , [ 2 , 4 , 1  ,5 ] , [ 0 , 3 , 5 ] ] #也就是说一共有6个单词
    N = 6#词典中单词个数
    n = 2#窗口数，这里一个窗口的长度是n+1
    m = 3#词向量维度
    #初始化词向量
    V = np.mat( [ [0.95583399,0.20597671, 0.35985563] , [0.52711512,0.69063238,0.10711258] , [0.14089753 ,0.15168923 ,0.98587405],[0.02302795 ,0.2735879 , 0.70455701] ,[0.75503672 ,0.96553685, 0.7692175 ],[0.81022171 ,0.23001825 ,0.20851955] ] )
    #这里使用固定的V进行测试
    hiden_num = 5 #隐层个数

    H = [[0.0413365 , 0.33844128, 0.15803457 ,0.60675624, 0.5136242 , 0.62602451],
 [0.92110622 ,0.81704241 ,0.13997044 ,0.70801429 ,0.43982256 ,0.23306125],
 [0.14744176 ,0.66998493 ,0.95154264 ,0.25288618 ,0.74008678 ,0.56844309],
 [0.44933381, 0.91515633 ,0.13544676, 0.9659042 , 0.26428839, 0.06309662],
 [0.14375444 ,0.7978239  ,0.15798142 ,0.4301479  ,0.19857439 ,0.8933008 ]]
    H = np.mat( H )

    d = [[0.4207357 ],
 [0.41657038],
 [0.31086192],
 [0.12878495],
 [0.50659333]]
    d = np.mat( d )

    W = [[0.05048497, 0.88955253, 0.36064096, 0.88605635, 0.64040486, 0.54561918],
         [0.10785829, 0.58053327, 0.95908276, 0.32916051, 0.213308, 0.82524347],
         [0.82763074, 0.12248368, 0.04700972, 0.65084266, 0.18883862, 0.60179424],
         [0.47286371, 0.76584485, 0.88442096, 0.63776017, 0.20073541, 0.00734413],
         [0.87597492, 0.56159428, 0.55996133, 0.26482261, 0.50497009, 0.75511432],
         [0.59966017, 0.76038106, 0.7645622, 0.71025932, 0.27069956, 0.89269468]]
    W = np.mat(W)

    U = [[0.95924224, 0.33825945, 0.03198492, 0.02356417, 0.69958963],
         [0.84190352, 0.89846008, 0.94832067, 0.58949293, 0.81342917],
         [0.793629, 0.35119279, 0.76382817, 0.96392716, 0.22060353],
         [0.09699234, 0.44392207, 0.6667028, 0.77008634, 0.01971949],
         [0.47857528, 0.86691483, 0.1289328, 0.34854161, 0.6994336],
         [0.69029762, 0.89215502, 0.40440472, 0.35761281, 0.82519782]]
    U = np.mat(U)

    b = [[0.94375993],
         [0.91319297],
         [0.23876379],
         [0.6872431],
         [0.60773373],
         [0.30955129]]
    b = np.mat(b)

    delta_b = np.mat( np.zeros( [ N , 1 ] ) )
    delta_U = np.mat( np.zeros( [ N , hiden_num ] ) )
    delta_W = np.mat( np.zeros( [ N , n * m ] ) )
    delta_d = np.mat( np.zeros( [ hiden_num , 1 ] ) )
    delta_H = np.mat( np.zeros( [ hiden_num , n * m ] ) )
    delta_x = np.mat( np.zeros( [1 , n * m ] ) )

    # d = np.mat( np.random.rand( hiden_num , 1 ) )
    # H = np.mat( np.random.rand( hiden_num , n * m ) )

    # V = np.mat( np.random.rand( N , m ) )
    # W = np.mat( np.random.rand( N , n * m ) )

    # U = np.mat( np.random.rand( N , hiden_num ) )


    # b = np.mat( np.random.rand( N , 1 ) )


    alpha = 0.1
    I = np.mat(np.ones( [ 1 , np.shape(W)[0] ] ))  # 形成一个矩阵规模为W的转置的元素全为1的矩阵

    g = 0

    for s in STN :
        for i in range(len(s) - n ) :
            #置0
            delta_d = np.mat(np.zeros([hiden_num, 1]))
            delta_H = np.mat(np.zeros([hiden_num, n * m]))
            delta_x = np.mat(np.zeros([1 , n * m]))


            #前向传播过程--已经验证
            x = V[ s[ i ] , : ].copy()#初始化x
            for j in range( 1 , n ) :
                x = np.hstack( ( x , V[ s[ i + j ] , : ] ) )
            #将前n-1个单词的向量进行拼接,最后得到的x是行向量
            h = tanh( d + H *x.T )#手动验证正确
            y = b + W * x.T + U * h

            #softmax--全连接--以验证正确性
            p_tempt = np.exp( y )
            p = p_tempt / p_tempt.sum()

            #反向传播
            delta_b = -p
            delta_b[ s[ i + n ] ] += 1
            for j in range( N ) :
                for q in range( n * m ) :
                    delta_W[ j , q ] = delta_b[ j ] * x[ 0 , q ]
                    delta_x[0,q] += delta_b[j] * W[ j , q ]

                for l in range( hiden_num ) :
                    delta_U[ j , l ] = delta_b[ j ] * h[ l ]
                    delta_d[l] += delta_b[j] * U[j, l] * (1 - h[l] ** 2)
                    for q in range( n * m ) :
                        delta_H[ l , q ] += delta_b[j] * U[ j , l ] * (1 - h[l] ** 2) * x[ 0 , q ]
                        delta_x[ 0 , q ] += delta_b[ j ] * U[ j , l ] * (1 - h[l] ** 2) * H[ l , q ]

            #更新
            b += alpha * delta_b
            W += alpha * delta_W
            U += alpha * delta_U
            d += alpha * delta_d
            H += alpha * delta_H
            x += alpha * delta_x
            for j in range( n ):
                V[s[i + j], :] = x[0 , range( m * j , m * ( j + 1 ) ) ].copy()


            # if g == 0 :
            #     print(b)
            #     print(W)
            #     print(U)
            #     print(d)
            #     print(H)
            #     print(x)
            #     print( V )
            #
            # g += 1
    print(V)

