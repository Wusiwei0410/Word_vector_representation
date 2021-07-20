#这个是在小规模数据上的测试函数———是主体部分
import numpy as np

def HardThah( x ) :
    #这里输入的是向量
    y = x.copy()
    y[ np.where( x < -1 ) ] = -1
    x[ np.where( x > 1) ] = 1
    return y

if __name__ == "__main__" :
    n = 6 #这个是词典中词的个数
    m = 3#词向量的长度
    hiden_num = 4 #隐藏层的个数
    windows = 3 #windows = 2 n + 1——这里的n不是字典个数，窗口中心词两边的词的个数
    alpha = 0.1

    #初始化
    # V = np.mat( np.random.rand( n , m ) )
    # W = np.mat( np.random.rand( hiden_num , m * windows ) )- 0.5
    # U = np.mat( np.random.rand( hiden_num , 1 ) )- 0.5
    # b = np.mat( np.random.rand( hiden_num , 1 ) )- 0.5
    Rand_num = [ 2 , 4 , 5 , 0 , 0 ]
    count = 0
    W = [[-0.14371492, -0.39029605, 0.07779875, 0.12131303, 0.00732542, 0.43672257,
          -0.37006025, 0.1867073, -0.40116372],
         [-0.48617401, 0.27306756, -0.12264465, -0.27773695, -0.49129108, -0.14743133,
          -0.21928619, -0.39986984, 0.35729733],
         [-0.24986201, -0.19395274, -0.0024916, 0.33942527, -0.14297808, 0.24258376,
          0.49610621, 0.22479849, -0.46464631],
         [0.0030376, 0.30220712, 0.48372307, -0.29361416, 0.03263739, -0.23855456,
          0.15903546, -0.00553383, 0.33731929]]
    W = np.mat(W)

    U = [[0.12072675],
         [0.4464694],
         [0.40359266],
         [0.2374443]]
    U = np.mat(U)

    b = [[0.08798889],
         [-0.38021725],
         [-0.24839792],
         [-0.28298028]]
    b = np.mat(b)

    #初始化小规模数据
    Doc = [[0, 1, 2, 3], [2, 4, 1, 5], [0, 3, 5]]  # 也就是说一共有6个单词
    # 初始化词向量
    V = np.mat([[0.95583399, 0.20597671, 0.35985563], [0.52711512, 0.69063238, 0.10711258],
                [0.14089753, 0.15168923, 0.98587405], [0.02302795, 0.2735879, 0.70455701],
                [0.75503672, 0.96553685, 0.7692175], [0.81022171, 0.23001825, 0.20851955]]) - 0.5

    for sentence in Doc :
        if len(sentence) >= windows :
            #只有在句子长度不小于窗口长度才进行传播
            for i in range( len(sentence) - windows + 1 ) :
                # 将当前窗口内单词提取出来，拼接成x
                x = V[sentence[ i ] , :].copy()
                for j in range( 1 , windows ) :
                    x = np.hstack( ( x , V[sentence[ i + j ] , :] ) )#最后得到的是行向量

                #生成随机样本进行对抗
                # rand_sample = np.random.randint( 0 , n ) #从0~n中生成随机整数
                # while rand_sample == sentence[ i + int( windows / 2 ) ] :
                #     #如果随机样本和窗口中心词相同，重新生成
                #     rand_sample = np.random.randint(0, n)  # 从0~n中生成随机整数
                # print( rand_sample )
                #这里的随机数需要给定
                rand_sample = Rand_num[ count ]
                count += 1

                #生成对抗x
                x_rand_sample = x.copy()
                x_rand_sample[ 0 , int( windows / 2 ) * m : ( int( windows / 2 ) + 1 ) * m ] = V[ rand_sample , :] #python中的int是向下取整的

                # 开始前向传播
                l = W * x.T + b
                h = HardThah( l )
                f = U.T * h
                l_rand_sample = W * x_rand_sample.T + b
                h_rand_sample = HardThah(W * x_rand_sample.T + b)
                f_rand_sample = U.T * h_rand_sample

                #反向传播
                if max( 0 , 1 - f + f_rand_sample ) > 0 :
                    #当loss function的值大于0时才反向传播
                    delta_U = h - h_rand_sample
                    one_l = np.mat(np.ones([hiden_num, 1]))
                    one_l[np.where(l < -1)] = 0
                    one_l[np.where(l > 1)] = 0
                    one_l_rand_sample = np.mat(np.ones([hiden_num, 1]))
                    one_l_rand_sample[np.where(l < -1)] = 0
                    one_l_rand_sample[np.where(l > 1)] = 0
                    delta_W = np.multiply( U , one_l ) * x - np.multiply( U , one_l_rand_sample ) * x_rand_sample
                    delta_b = np.multiply( U , one_l - one_l_rand_sample )
                    delta_x = np.multiply( U , one_l ).T * W
                    delta_x_rand_sample = np.multiply( U , one_l_rand_sample ).T * W

                    #更新
                    U += alpha * delta_U
                    W += alpha * delta_W
                    b += alpha * delta_b
                    for j in range( windows ):
                        if j != int( windows / 2 ) :
                            #更新x和x'中相同的部分
                            V[ sentence[ i + j ] ] += alpha * ( delta_x[ 0 , j * m : ( j + 1 ) * m ] - delta_x_rand_sample[ 0 , j * m : ( j + 1 ) * m ] )
                        else :
                            V[sentence[i + j]] += alpha * delta_x[0, j * m: (j + 1) * m]
                            V[rand_sample] -= alpha * delta_x_rand_sample[ 0 , j * m : ( j + 1 ) * m ]
                print( V )