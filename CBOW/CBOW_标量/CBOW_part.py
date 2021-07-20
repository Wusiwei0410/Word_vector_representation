#这个是CBOW在小规模数据上的测试函数
import  numpy as np

if __name__ == "__main__" :
    n = 6  # 这个是词典中词的个数
    m = 3  # 词向量的长度
    hiden_num = 4  # 隐藏层的个数
    windows = 3  # windows = 2 n + 1——这里的n不是字典个数，窗口中心词两边的词的个数
    C = int( windows / 2 ) #中心词两侧窗口长度——python是向下取整的
    alpha = 0.1

    # 初始化
    # V = np.mat( np.random.rand( n , m ) )
    W = np.mat(np.random.rand(hiden_num, m * windows)) - 0.5
    U = np.mat(np.random.rand(hiden_num, 1)) - 0.5
    b = np.mat(np.random.rand(hiden_num, 1)) - 0.5

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
                delta_V = np.mat( np.zeros([ n , m ]) )
                sum_V = np.mat( np.zeros( [ 1 , m ] ) )

                L = []
                for j in range( windows ) :
                    L.append( sentence[ i + j ] )#记录当前窗口下所有词的下标
                x = (sum(V[ L ]) - V[ L[ C ] ] ) / ( windows - 1 ) #计算平均值_行向量

                #前向传播
                y = V * x.T
                p = np.exp( y ) / np.exp( y ).sum()

                #反向传播
                delta_y = -p
                delta_y[ L[ C ] ] += 1

                #先计算x的导数，也就是sum_V
                for j in range( m ) :
                    for q in range( n ) :
                        sum_V[ 0 , j ] += delta_y[ q ] * V[ q , j ]

                for q in range( n ) :
                    if q in L :
                        #如果当前窗口中包含当前行对应的单词
                        #查看当前窗口中该单词的个数，并进行对应的常系数
                        tempt = L.count(q) / (2 * C) #底数—-count(q)/2n
                        for j in range(m):
                            delta_V[q, j] = delta_y[q] * x[0, j] + sum_V[0, j]*tempt
                    else :
                        for j in range(m):
                            delta_V[q, j] = delta_y[q] * x[0, j]

                #更新
                V += alpha * delta_V

    print( V )
