#这里是用来测试的程序
import  numpy as np

if __name__ == "__main__" :
    n = 6  # 这个是词典中词的个数
    m = 3  # 词向量的长度
    hiden_num = 4  # 隐藏层的个数
    windows = 3  # windows = 2 n + 1——这里的n不是字典个数，窗口中心词两边的词的个数
    C = int( windows / 2 ) #中心词两侧窗口长度——python是向下取整的
    alpha = 0.1

    # 初始化
    W = [[-4.65486908e-02, -2.55160778e-02, -3.43973387e-02,  3.81992571e-02,
   4.33684287e-02,-1.38432548e-02 , 2.61977304e-02, -4.15780771e-02,
   3.85014432e-02] ,
 [-4.10407903e-02 ,-4.39490956e-02 ,-6.28422174e-03  ,3.54492792e-02,
   3.48985833e-02  ,2.54855271e-02, -4.59307629e-02, -1.54518628e-02,
  -3.04710905e-02],
 [-3.15595084e-02 , 1.08666662e-02 ,-4.79970068e-02, -1.02137750e-02,
   1.17519003e-02  ,1.81797861e-02  ,2.21830455e-02, -2.48284984e-02,
  -1.34096822e-03],
 [-3.78331968e-02 ,-1.20648005e-02 ,-4.58054208e-02 , 1.35308935e-02,
   4.48894009e-05 , 3.08864803e-02 , 2.63042164e-02 ,-1.29336112e-02,
   1.86406782e-02]]

    b = [[0.01543685],
 [0.04394029],
 [0.00026561],
 [0.01413836]]

    U = [[-0.02560679],
 [-0.03505162],
 [-0.00288087],
 [ 0.00834527]]
    # V = np.mat( np.random.rand( n , m ) )
    # W = np.mat(np.random.rand(hiden_num, m * windows)) / 10 - 0.05
    # U = np.mat(np.random.rand(hiden_num, 1)) / 10 - 0.05
    # b = np.mat(np.random.rand(hiden_num, 1)) / 10 - 0.05

    # 初始化小规模数据
    STN = [[0, 1, 2, 3], [2, 4, 1, 5], [0, 3, 5]]  # 也就是说一共有6个单词
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

                L = []
                for j in range(windows):
                    L.append(sentence[i + j])  # 记录当前窗口下所有词的下标

                #前向传播
                y = V * x.T
                p = np.exp( y ) / np.exp( y ).sum()

                #反向传播
                delta_V = np.mat( np.zeros( [ n , m ] ) )#初始化delta_V
                for j in range( windows ) :
                    if j != C :
                        #以中心词来估计的窗口中其他词出现概率最大为目标
                        delta_y = -p#记录当前单词的onehot概率
                        delta_y[ L[ j ] ] += 1

                        sum_V = np.mat(np.zeros([1, m]))  # 初始化sum_V
                        for k in range(m):
                            for q in range(n):
                                sum_V[0, k] += delta_y[q] * V[q, k]
                        for q in range( n ) :
                            for k in range( m ) :
                                delta_V[ q , k ] += delta_y[ q ] * x[ 0 , k ]
                        for k in range( m ) :
                            delta_V[ L[ j ] , k ] +=  sum_V[ 0 , k ]
                # 更新
                V += alpha * delta_V

    print( V )
