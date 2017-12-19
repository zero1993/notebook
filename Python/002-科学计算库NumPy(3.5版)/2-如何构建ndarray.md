## 如何构建ndarray

#### array构建

- 使用array函数构建ndarray
    - 使用numpy.array函数将list转换为ndarray类型的向量/矩阵
```
import numpy as np
vector = np.array([1,2,3,4])
print (vector)
matrix = np.array([[11,12,13],[21,22,23],[31,32,33]])
print (matrix)
```
==>
```
[1 2 3 4]
[[11 12 13]
 [21 22 23]
 [31 32 33]]
```

- 元素类型一致性
    - ndarray中的数据类型保持一致
    - numpy将自动转换所有的值，使类型一致
    - 可以通过dtype属性查看
    - 传入4个int，dtype为int，当有一个元素为float是所有元素自动转换
    - 使用astype函数可以进行整体类型转换
```
import numpy as np
vector = np.array([1,2,3,4.0])
print (vector.dtype) # ==> float64
vector = np.array([1,2,3])
print (vector.dtype) # ==> int32
vector = vector.astype(float)
print (vector.dtype) # ==> float64
```

- shape查看向量/矩阵的结构
    - 使用shape属性查看
```
import numpy as np
vector = np.array([1,2,3,4])
print (vector.shape) # ==> (4,)
matrix = np.array([[11,12,13],[21,22,23],[31,32,33]])
print (matrix.shape) # ==> (3, 3)
```

- 按索引取值
    - ndarray数据结构支持使用索引取值
    - 支持python中的切片
```
import numpy as np
vector = np.array([1,2,3,4])
print (vector[1])
print (vector[0:2])
matrix = np.array([[11,12,13],[21,22,23],[31,32,33]])
print (matrix[1,1])
print (matrix[0:2,0:2])
```
==>
```
2
[1 2]
22
[[11 12]
 [21 22]]
```

#### arange快速构建

- 通过arange快速构造一维的ndarray
    - 方式1：传入单个参数n，创建从零开始到n-1的数据集，可传入小数
    - 方式2：传入两个参数x,y，指定范围x~y-1
    - 方式3：传入三个参数，类似于python切片
```
import numpy as np
# 方式1
a = np.arange(10)
print (a) # ==> [0 1 2 3 4 5 6 7 8 9]
a = np.arange(6.3)
print (a) # ==> [ 0.  1.  2.  3.  4.  5.  6.]
# 方式2
a = np.arange(1, 6)
print (a) # ==> [1 2 3 4 5]
a = np.arange(1.1, 6.8)
print (a) # ==> [ 1.1  2.1  3.1  4.1  5.1  6.1]
# 方式3，有点类似切片
a = np.arange(0, 10, 2)
print (a) # ==> [0 2 4 6 8]
a = np.arange(0,5,0.7)
print (a) # ==> [ 0.   0.7  1.4  2.1  2.8  3.5  4.2  4.9]
```

- 通过reshape将arange创建的一维变为多维
    - 传入参数x,y，指定二维数据的维度，确定其中一个参数，另一个可以传入-1，自动计算
    - 传入n个参数，指n维数据的维度，确定其中n-1个参数，另一个可以传入-1自动计算
```
import numpy as np
a = np.arange(10)
print (a)
print (a.shape)
a = a.reshape(2, 5)
print (a)
print (a.shape)
a = a.reshape(5, -1)
print (a)
print (a.shape)
```
==>
```
[0 1 2 3 4 5 6 7 8 9]
(10,)
[[0 1 2 3 4]
 [5 6 7 8 9]]
(2, 5)
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
(5, 2)
```

- 通过ravel将多维变为一维
    - 可以使用ndarray对象的ravel()函数
    - 也可以使用numpy的ravel函数
    - 使用reshape(-1)来进行此操作
```
import numpy as np
a = np.array([[1,2],
              [4,5]])
print (a)
print (a.shape)
a = np.ravel(a) # 或 a = a.ravel() 或  a = a.reshape(-1)
print (a)
print (a.shape)
```
==>
```
[[1 2]
 [4 5]]
(2, 2)
[1 2 4 5]
(4,)
```

- 通过ndim查看ndarray的维度
    - 打印ndarray对象的ndim属性，可以查看维度
    - 使用numpy的ndim函数也可以查看维度
    - 标量的ndim为0
```
import numpy as np
a = np.arange(10)
print (a.ndim) # ==> 1
a = a.reshape(2, 5)
print (np.ndim(a)) # ==> 2
print (np.ndim(1)) # ==> 0
```

- 通过size查看ndarray的元素个数
    - 打印ndarray对象的size属性
    - 使用numpy的size函数
    - size函数支持传入第二个参数0/1表示按列/行统计size
```
import numpy as np
a = np.arange(10).reshape(2, 5)
print (a.size) # ==> 10
print (np.size(a)) # ==> 10
print (np.size(a, 0)) # ==> 2
print (np.size(a, 1)) # ==> 5
```

#### 其他方式构建

- 使用zeros&ones构建固定数据矩阵
    - 传入int参数，创建一维ndarray，数据值均为0
    - 传入包含n个参数的tuple，创建n维ndarray，数据均为0
    - 数据值类型默认为浮点型，传入可选参数dtype修改数据类型
```
import numpy as np
# 传入int创建
a = np.zeros(5)
print (a) # ==> [ 0.  0.  0.  0.  0.]
# 传入tuple创建
a = np.zeros((2, 3))
print (a) # ==> [[ 0.  0.  0.][ 0.  0.  0.]]
# 执行dtype为int类型
a = np.zeros((2, 3), dtype=np.int)
print (a) # ==> [[0 0 0][0 0 0]]
# 使用ones()构建元素值为1的ndarray，具体用法同zeros
a = np.ones(5)
print (a) # ==> [ 1.  1.  1.  1.  1.]
```

- random构建
    - 使用np.random模块下的random方法创建随机数据，范围0.0~1.0
    - python2.7中仅提供了random_sample方法
    - 无参数调用，返回随机标量
    - int参数调用，返回一维数据
    - tuple调用，返回n维数据
```
import numpy as np
# 无参调用
a = np.random.random_sample()
print a
# int参调用
a = np.random.random_sample(2)
print a
# 传入tuple调用
a = np.random.random_sample((3,4))
print a
```
==>
```
0.8565811279533561
[ 0.85849862  0.72249025]
[[ 0.36489648  0.03103976  0.06647924  0.50873314]
 [ 0.6571564   0.23151547  0.01842436  0.60679232]
 [ 0.43173448  0.61011263  0.89970167  0.67587247]]
```

- 使用np.linspace创建范围内的均值
    - 前两个参数范围，第三个参数指定取值数量，不传入时默认为50
```
import numpy as np
# 开始值0，结束值5，平均取6个数
a = np.linspace(0,5,6)
print (a) # ==> [ 0.  1.  2.  3.  4.  5.]
```

- 文件读取构建
    - genfromtxt函数可以从txt中读取数据
    - 参数以表示文件路径，示例代码使用相对路径，py文件同目录，delimiter制定数据之间的分隔符，dtype指定读取的类型
```
import numpy as np
txtdata = np.genfromtxt('txtdata.txt',delimiter=',',dtype=str)
```
