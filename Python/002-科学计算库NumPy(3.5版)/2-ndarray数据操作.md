## ndarray数据操作

#### 运算操作

- 逻辑运算
    - 使用==判断ndarray中是否包含某值，返回值是元素为boolean类型的ndarray
    - 返回的值可以充当索引，进行取值
    - 返回值可以进行与和或的运算
```
import numpy as np
vector = np.array([1,2,3,4])
result = (vector == 3)
print (type(result)) # ==> <class 'numpy.ndarray'>
print (result) # ==> [False False  True False]
print (vector[result]) # ==> [3]
result = (vector == 1) & (vector ==2)
print (result) # ==> [False False False False]
result = (vector ==1) | (vector == 2)
print (result) # ==> [ True  True False False]
```

- 基本算数运算
    - 列举符号运算
    - 表示矩阵对应位置相乘，矩阵乘法使用dot函数
```
import numpy as np
a = np.array([[1,0],
              [3,4]])
print (a)
# 所有数据-1
print (a-1)
# 所有数据+1
print (a+1)
# 所有数据乘方
print (a**2)
# 所有数据判断
print (a<2)
b = np.array([[2,3],
              [0,5]])
print (b)
# 对应位置相加
print (a+b)
# 对应位置相乘
print (a*b)
# 矩阵乘法
print (a.dot(b))
print (np.dot(a,b))
```
==>
```
[[1 0]
 [3 4]]
[[ 0 -1]
 [ 2  3]]
[[2 1]
 [4 5]]
[[ 1  0]
 [ 9 16]]
[[ True  True]
 [False False]]
[[2 3]
 [0 5]]
[[3 3]
 [3 9]]
[[ 2  0]
 [ 0 20]]
[[ 2  3]
 [ 6 29]]
[[ 2  3]
 [ 6 29]]
```

- 求极值
    - 使用max/min函数求最大/小值
    - 不传入参数返回数据集的最大/最小值，dtype即为数据的类型
    - 可以传入axis=0/1可选参数，按列/行取最大/小值，返回值仍为ndarray
```
import numpy as np
vector = np.array([1,2,3])
print (vector.min()) # ==> 1
matrix = np.array([[2,3],
                   [1,9]])
print (matrix.min()) # ==> 1
print (matrix.min(axis=1)) # ==> [2 1]
print (matrix.max(axis=0)) # ==> [2 9]
```

- 求和操作
    - 矩阵可以使用sum函数进行求和
    - 不传入参数返回数据集求和值，dtype即为数据的类型
    - 可以传入axis=0/1可选参数，按列/行求和，返回值仍为ndarray
```
import numpy as np
matrix = np.array([[1,2,5],
                   [7,4,1],
                   [2,2,2]])
print (matrix.sum()) # ==> 26
print (matrix.sum(axis=1)) # ==> [8 12 6]
print (matrix.sum(axis=0)) # ==> [10 8 8]
```

- 求通过T属性逆矩阵
    - 使用对象的T属性
```
import numpy as np
a = np.arange(6).reshape(3,-1)
print (a)
print (a.T)
```
==>
```
[[0 1]
 [2 3]
 [4 5]]
[[0 2 4]
 [1 3 5]]
```

- 使用sqrt计算开方
    - 使用sqrt函数可以计算数值的开方值
    - 可以传入ndarray
```
import numpy as np
a = np.arange(5)
print (a) # ==> [0 1 2 3 4]
print (np.sqrt(4)) # ==> 2.0
print (np.sqrt(a)) # ==> [ 0.          1.          1.41421356  1.73205081  2.        ]
```

- 使用floor函数进行向下取整操作，即取出小数部分
    - 取出数值的小数部分
    - 可传入ndarray
```
import numpy as np
a = 10*np.random.random_sample((3,4))
print (a)
a = np.floor(a)
print (a)
print (np.floor(2.6))
```
==>
```
[[ 7.22985559  3.71012084  6.53118107  4.50488865]
 [ 8.27915719  3.11237321  2.07708957  1.18242878]
 [ 4.17740903  1.79049349  8.76313204  6.76453788]]
[[ 7.  3.  6.  4.]
 [ 8.  3.  2.  1.]
 [ 4.  1.  8.  6.]]
2.0
```

- 使用exp计算e的n次幂
    - 使用exp函数可以计算e的n次幂
    - 可以传入ndarray
```
import numpy as np
a = np.arange(3)
print (a) # ==> [0 1 2]
print (np.exp(1)) # ==> 2.71828182846
print (np.exp(a)) # ==> [ 1.          2.71828183  7.3890561 ]
```

#### 结构操作

- reshape&reval参考《2-如何构建ndarray》

- 使用hstack/vstack进行横向/纵向矩阵拼接操作
```
import numpy as np
a = np.floor(10*np.random.random_sample((2,2)))
b = np.floor(10*np.random.random_sample((2,2)))
print (a)
print (b)
print (np.hstack((a, b)))
print (np.vstack((a, b)))
```
==>
```
[[ 5.  4.]
 [ 1.  3.]]
[[ 8.  6.]
 [ 1.  9.]]
[[ 5.  4.  8.  6.]
 [ 1.  3.  1.  9.]]
[[ 5.  4.]
 [ 1.  3.]
 [ 8.  6.]
 [ 1.  9.]]
```

- 使用hsplit/vsplit进行横向/纵向矩阵切分操作
    - 传入int平均等分
    - 传入tuble，指定位置切分
```
import numpy as np
a = np.arange((18)).reshape(-1,6)
print (a)
print (np.hsplit(a,3))
print (np.hsplit(a,(2,)))
print (np.hsplit(a,(1,3,5)))
```
==>
```
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]]
[array([[ 0,  1],
       [ 6,  7],
       [12, 13]]), array([[ 2,  3],
       [ 8,  9],
       [14, 15]]), array([[ 4,  5],
       [10, 11],
       [16, 17]])]
[array([[ 0,  1],
       [ 6,  7],
       [12, 13]]), array([[ 2,  3,  4,  5],
       [ 8,  9, 10, 11],
       [14, 15, 16, 17]])]
[array([[ 0],
       [ 6],
       [12]]), array([[ 1,  2],
       [ 7,  8],
       [13, 14]]), array([[ 3,  4],
       [ 9, 10],
       [15, 16]]), array([[ 5],
       [11],
       [17]])]
```

- numpy中的复制操作
    - 用=直接赋值，并不是赋值，两个引用指向同一内存
    - view()浅复制：复制出一个新的对象，和原来的对象共用数据值，不共用结构
```
import numpy as np
# 赋值
a = np.arange(12).reshape(2,6)
b = a
print (b is a) # ==> True
b.shape = (3,4)
print (a.shape) # ==> (3, 4)
# 浅复制
c = a.view()
print (c is a) # ==> False
c.shape = (4,3)
print (a.shape) # ==> (3, 4)
c[0,1] = 9999
print (a[0,1]) # ==> 9999
# 复制
d = a.copy()
print (d is a) # ==> False
d.shape = (6,2)
print (a.shape) # ==> (3 ,4)
d[0,1] = 7777
print (a[0,1]) # ==> 9999
```

- 求矩阵中的最大值
    - 使用argmax函数找到最大值索引
    - 可传入参数axis指定按行/列
```
import numpy as np
data = np.sin(np.arange(20)).reshape(5,4)
print (data)
# 找到整体数据最大值索引，按一维数据索引
print (data.argmax())
# 找到每列的最大值索引
ind = data.argmax(axis=0)
print (ind)
# 通过索引依次打印每列最大值
data_max = data[ind,range(data.shape[1])]
print (data_max)
# 找到每行的最大值索引
ind = data.argmax(axis=1)
print (ind)
data_max = data[range(data.shape[0]),ind]
print (data_max)
```
==>
```
[[ 0.          0.84147098  0.90929743  0.14112001]
 [-0.7568025  -0.95892427 -0.2794155   0.6569866 ]
 [ 0.98935825  0.41211849 -0.54402111 -0.99999021]
 [-0.53657292  0.42016704  0.99060736  0.65028784]
 [-0.28790332 -0.96139749 -0.75098725  0.14987721]]
14
[2 0 3 1]
[ 0.98935825  0.84147098  0.99060736  0.6569866 ]
[2 3 0 2 3]
[ 0.90929743  0.6569866   0.98935825  0.99060736  0.14987721]
```

- 使用np.tile对矩阵进行扩展
    - 指定int参数，按横向扩展n倍
    - 指定tuple参数，按维度扩展
```
import numpy as np
a = np.arange(0,40,10).reshape(2,2)
print (a)
# 横向扩展2倍
print (np.tile(a,2))
# 行扩展4倍，列扩展三倍
print (np.tile(a,(4, 3)))
```
==>
```
[[ 0 10]
 [20 30]]
[[ 0 10  0 10]
 [20 30 20 30]]
[[ 0 10  0 10  0 10]
 [20 30 20 30 20 30]
 [ 0 10  0 10  0 10]
 [20 30 20 30 20 30]
 [ 0 10  0 10  0 10]
 [20 30 20 30 20 30]
 [ 0 10  0 10  0 10]
 [20 30 20 30 20 30]]
```

- 矩阵排序
    - 使用sort函数对数据进行排序
    - 使用np.sort函数对数据进行排序
    - 使用argsort函数得到排序索引
```
import numpy as np
# 使用对象的sort函数，将对象转换为排序后的对象，注意，这里方法调用的返回值为None
a = np.array([[6,1,9],[4,3,8]])
print (a)
a.sort(axis=1)
print (a)
a.sort(axis=0)
print (a)
# 使用np.sort函数，将转换后的对象作为函数的返回值
b = np.array([[6,1,9],[4,3,8]])
print (b)
print (np.sort(b,axis=0))
# 使用argsort得到排序索引值的矩阵,并可通过得到的索引值打印出排序后的矩阵
c = np.array([4,2,1,3])
index = np.argsort(c)
print (index)
print (c[index])
```
==>
```
[[6 1 9]
 [4 3 8]]
[[1 6 9]
 [3 4 8]]
[[1 4 8]
 [3 6 9]]
[[6 1 9]
 [4 3 8]]
[[4 1 8]
 [6 3 9]]
[2 1 3 0]
[1 2 3 4]
```
