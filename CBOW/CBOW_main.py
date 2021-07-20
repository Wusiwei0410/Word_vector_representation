#因为CBOW也是采用的n=2C+1的窗口来预测中间词的概率，并以句子为输入，这里就调用了RNN的命令
import FNN神经网络语言模型__NNLM.NNML_name as NNML
import numpy as  np

def CBOW( W , V , C  ) :
    alpha = 0.1
    windows = 2 * C + 1#窗口大小
    n = len( V ) #词典中单词个数
    m = len( V.T ) #向量空间的维度
    U = np.mat(np.random.rand( n , m ))
    for statement in W :
        #提取句子
        if len(statement) >= windows :
            #只有在窗口数不大于句子长度时才能进行训练
            for i in range( len(statement) - windows + 1 ) :
                #提取x
                x = np.mat(np.zeros([m,1]))
                for j in range( windows ) :
                    if j != C :
                        #去除掉中间词汇
                        x += V[ statement[i + j] , : ].T
                #取平均值
                x = x / ( windows - 1 )

                #前向传播
                y = np.dot( V , x )
                p = np.exp( y )
                p = p / p.sum()

                #反向传播
                delta_y = -p
                delta_y[ statement[ i + C ] ] = 1 + delta_y[ statement[ i + C ] ]
                delta_x = np.dot( V.T , delta_y )
                delta_V = U
                for j in range( n ) :
                    delta_V[ i , : ] = delta_y[ i ] * x.T

                V += alpha * delta_V
                for j in range(windows):
                    if j != C:
                        # 去除掉中间词汇
                        V[statement[i + j] , : ] += alpha * delta_x.T / windows
    return V

if __name__ == "__main__" :
    print("*==================================开始==========================================*")
    L = NNML.stop_set("E:\作业\data\stopwords-master\cn_stopwords.txt")  # 提取停词表
    path = "E:/作业/data"
    file_name = path + "/data.txt"
    W = NNML.data_set(path, L)  # 去停词化和分词化
    print(W)
    Dic = NNML.Dic_set(W)  # 建立词典
    W = NNML.change(W, Dic)  # 将W转换为Dic中对应的序号表示
    n = len( Dic )#词表中单词个数

    V = np.mat( np.random.uniform( - 1 ,  1 ,  (n , 5) ) )  #向量空间是5维的
    C = 3 #用来确定窗口数Windows=2C+1
    V = CBOW( W , V , C  )
    print( W )
    print( V )
