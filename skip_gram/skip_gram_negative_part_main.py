#这里是用来测试的程序
import  numpy as np
import random
import math

def calculate_p( Fre ) :
    tempt = []
    p = []
    sum = 0
    for i in Fre :
        sum += i**0.75
        tempt.append(sum)

    M = max(tempt)
    #将p归一化
    for i in tempt :
        p.append(i / M)

    return  p

def find_index( x , p ) :
    #根据生成的0-1之间的随机数，找到对应的下标
    for i in range( len( p ) ) :
        if p[ i ] > x :
            return i

def rand_k( k , p1 , p2 , p ) :
    #产生k个0-1之间的随机数
    L = []
    tempt = []
    while len(L) < k :
        print( "输出:" )
        print(L)
        print( tempt )

        #产生一个随机数
        r = random.uniform( 0 , 1 )
        print(r , p1 , p2)
        if r not in L and (r > p2 or r < p1):
            tempt.append( r )
            L.append( find_index( r , p ) )
    return L

if __name__ == "__main__" :
    n = 6  # 这个是词典中词的个数
    m = 3  # 词向量的长度
    hiden_num = 4  # 隐藏层的个数
    windows = 3  # windows = 2 n + 1——这里的n不是字典个数，窗口中心词两边的词的个数
    negative_num = 5 #负采样的个数
    C = int( windows / 2 ) #中心词两侧窗口长度——python是向下取整的
    alpha = 0.1

    # 初始化
    W = [[-4.65486908e-02, -2.55160778e-02, -3.43973387e-02, 3.81992571e-02,
          4.33684287e-02, -1.38432548e-02, 2.61977304e-02, -4.15780771e-02,
          3.85014432e-02],
         [-4.10407903e-02, -4.39490956e-02, -6.28422174e-03, 3.54492792e-02,
          3.48985833e-02, 2.54855271e-02, -4.59307629e-02, -1.54518628e-02,
          -3.04710905e-02],
         [-3.15595084e-02, 1.08666662e-02, -4.79970068e-02, -1.02137750e-02,
          1.17519003e-02, 1.81797861e-02, 2.21830455e-02, -2.48284984e-02,
          -1.34096822e-03],
         [-3.78331968e-02, -1.20648005e-02, -4.58054208e-02, 1.35308935e-02,
          4.48894009e-05, 3.08864803e-02, 2.63042164e-02, -1.29336112e-02,
          1.86406782e-02]]

    b = [[0.01543685],
         [0.04394029],
         [0.00026561],
         [0.01413836]]

    U = [[-0.02560679],
         [-0.03505162],
         [-0.00288087],
         [0.00834527]]
    # V = np.mat( np.random.rand( n , m ) )
    # W = np.mat(np.random.rand(hiden_num, m * windows)) / 10 - 0.05
    # U = np.mat(np.random.rand(hiden_num, 1)) / 10 - 0.05
    # b = np.mat(np.random.rand(hiden_num, 1)) / 10 - 0.05

    # 初始化小规模数据
    STN = [[0, 1, 2, 3], [2, 4, 1, 5], [0, 3, 5]]  # 也就是说一共有6个单词
    Fre = [2, 2, 2, 2, 1, 2]  # 频率列表——节点权重

    #根据词频指定采样概率
    p = calculate_p( Fre ) #根据采样频率计算概率分布
    print( p , find_index( 0.7 , p ) )

    # 初始化词向量
    V = np.mat([[0.95583399, 0.20597671, 0.35985563], [0.52711512, 0.69063238, 0.10711258],
                [0.14089753, 0.15168923, 0.98587405], [0.02302795, 0.2735879, 0.70455701],
                [0.75503672, 0.96553685, 0.7692175], [0.81022171, 0.23001825, 0.20851955]]) - 0.5

    print( V )

    for sentence in STN :
        if len(sentence) >= windows :
            #如果句子长度不小于窗口长度
            for i in range(len(sentence) - windows + 1) :
                x = V[ sentence[ i + C ] , : ].copy() #窗口中心词向量为x
                delta_V = np.mat( np.zeros( [ n , m ] ) ) #初始化导数
                delta_x = np.mat( np.zeros( [ 1 , m ] ) )
                for j in range( windows ) :
                    if j != C :
                        #对于上下文都产生负采样
                        if sentence[ i + j ] == 0 :
                            rand_num = rand_k(negative_num, 0 , p[ sentence[ i + j  ]  ] , p )
                        else:
                            rand_num = rand_k(negative_num, p[sentence[i + j ] - 1 ] , p[sentence[i + j ]] , p)
                        print( rand_num )
                        for word in rand_num :
                            delta_V[ word , : ] -= 1 / ( 1 + math.exp( (-x * V[ word , : ].T)[0 , 0] ) ) * x
                            delta_x -= 1 / ( 1 + math.exp( (-x * V[ word , : ].T)[0 , 0] ) ) * V[ word , : ]
                        delta_V[sentence[i + j ], :] += ( 1 - 1 / (1 + math.exp((-x * V[sentence[i + j ], :].T)[0,0]))) * x
                        delta_x += ( 1 - 1 / (1 + math.exp((-x * V[sentence[i + j ], :].T)[0,0]))) * V[sentence[i + j ], :]
                #更新
                V += alpha * delta_V
                V[sentence[i + C], :] += alpha * delta_x
    print( V )
