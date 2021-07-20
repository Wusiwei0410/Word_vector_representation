import numpy as np
import  math

def find_max( F ) :
    #这个程序时找出列表F中最大的值
    Max = -100#初始化一个很小的值
    for i in range( len(F) ) :
        if F[i] > Max :
            Max = F[i]
            index =  i
    return Max , index

def sort( F ) :
    #这个函数是对列表F进行排序
    #将排序后的列表，以及对应之前列表中的index列表输出
    R = F[:]
    sort_F = []
    Index = []
    for i in range( len( R ) ) :
        Max , index = find_max( R )
        #将结果放入列表
        sort_F.append( Max )
        Index.append( index )

        R[ index ] = -100 #以及找过的数，给予一个很小的值

    return sort_F , Index

def Delete_end_2( F ) :
    F.remove(F[len(F) - 1])
    F.remove(F[len(F) - 1])

def fusion( Node , Index , Fre , count , n ) :
    #将栈中最小的两个节点融合删除后，产生新的节点，并将新节点的值和下标index放入栈，
    power = Fre[ len( Fre ) - 1 ] + Fre[ len( Fre ) - 2 ]
    Node[ Index[ len( Index ) - 1 ] ][ 0 ] = n + count
    Node[ Index[ len( Index ) - 2 ] ][ 0 ] = n + count
    Node.append( [ None , power , Index[ len( Index ) - 1 ] , Index[ len( Index ) - 2 ] , n + count] )
    Delete_end_2( Index )
    Delete_end_2(Fre)

    #直接往后搜索，按照新节点的权值插入栈，并放入下标index
    if len(Fre) == 0:
        #如果恰好为空
        Fre = [power] + Fre
        Index = [count + n] + Index
    else :
        for i in range(len(Index) - 1, -1, -1):
            if power > Fre[i]:
                if i == 0:
                    #当迭代到第一个元素，power依旧比它大
                    Fre = [power] + Fre
                    Index = [count + n] + Index
            else:
                #当power比当前数小
                Fre = Fre[0: i + 1] + [power] + Fre[i + 1: len(Fre)] #这种拼接之后，函数里的列表地址可能发生了改变，不在影响主函数里的列表值
                Index = Index[0: i + 1] + [n + count] + Index[i + 1: len(Fre)]

    count = count + 1
    return  Fre , Index , count , Node

if __name__ == "__main__" :
    # 初始化小规模数据
    STN = [[0, 1, 2, 3], [2, 4, 1, 5], [0, 3, 5]]  # 也就是说一共有6个单词
    windows = 1 #单侧窗口的长度
    alpha = 0.1
    Fre = [ 2 ,2 ,2 ,2 ,1 ,2 ] #频率列表——节点权重
    n = len(Fre)

    #首先，为每一个已知的单词建立节点
    Node = []
    for i in range( n ) :
        Node.append( [ None , Fre[ i ] , None , None , i ] )
        # 每个节点中包括的内容分别是
        #(父节点 ， 权值 ， 左节点 ， 右节点 , 节点本身的index )

    Fre , Index = sort(Fre)#排序

    count = 0 #记录生成的非叶子节点的个数
    while len( Fre ) != 1 :
        #当只有一个节点时，迭代停止
        Fre , Index , count , Node = fusion( Node , Index , Fre , count , n ) #融合栈中最小的两个节点，形成新节点

    #生成所有叶子结点对应的哈夫曼编码，以及路径上的节点的下标
    HF_code = [] #记录哈夫曼编码
    L_path = [] #记录叶子结点路径上的非叶子节点
    for i in range( 0 , n ) :
       N = Node[ i ]
       code = []
       P = []
       while 1 :
           # 直到搜索到根节点则跳出
           if N[0] == None:
               break
           P = [N[ 0 ]] + P
           if Node[ N[ 0 ] ][ 2 ] == N[ 4 ] :
               #左节点则是1
               code = [ 1 ] + code
           else :
               # 左节点则是0
               code = [0] + code
           N = Node[ N[ 0 ] ]
       L_path.append(P)
       HF_code.append(code)
    print( L_path )
    print( HF_code )

    theta = np.mat( np.random.rand( n - 1 , n - 1 ) ) #非叶子节点的向量
    A = np.mat( np.random.rand( n , n ) ) #叶子节点的向量
    for segment in STN :
        if len( segment ) > windows * 2 :
            #生成delta变量
            delta_theta = np.mat( np.zeros( [n - 1 , n - 1] ) )
            delta_x = np.mat( np.zeros( [n , 1] ) )
            #输入的片段的长度不能小于窗口的长度
            for i in range( len( segment ) - windows * 2 ) :
                # 先均值化生成x
                x = np.mat(np.zeros([n, 1]))
                for j in range( windows * 2 + 1  ) :
                    if windows != j :
                        #去除窗口中心需要预测的单词
                        x += A[segment[i + j], :].T
                x = x / ( 2 * windows )
                code = HF_code[segment[ i + windows ]] #取出哈夫曼编码
                P = L_path[segment[ i + windows ]]
                for j in range( len( P ) ) :
                    delta_theta[ P[ j ] , : ] += alpha * ( code[ j ] - 1 / ( 1 + math.exp( - x.T * theta[P[ j ] , :] ) ) ) * x.T
                    delta_x += alpha * alpha * ( code[ j ] - 1 / ( 1 + math.exp( - x.T * theta[P[ j ] , :].T ) ) ) * theta[P[ j ] , :].T

                # 更新
                theta += delta_theta
                for j in range( windows * 2 + 1 ) :
                    if j != windows :
                        A[ segment[ i + j ] ] += delta_x / ( windows * 2 )

    print( A )
    print( theta )