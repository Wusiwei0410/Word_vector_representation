# import numpy as np
#
# def give_value( A  , B ) :
#     #将B中的值赋给A
#     n , m = np.shape( A ) ##读取矩阵的维数
#     if n == 1 | m == 1 :
#         #如果是向量的话
#         n = max( m , n )
#         for i in range( n ) :
#             A[i] = B[i]#进行向量赋值
#     else :
#         for i in range( n ) :
#             for j in range( m ) :
#                A[ i , j ] = B[ i , j ]
#     return  A
#
# if __name__ == "__main__" :
#     x = np.mat('1 2 3 ; 7 8 9')
#     y = 1 * x[ 0  , :  ].T
#     y[ 0 , 0] += 1
#     print( x )
#     print( y )