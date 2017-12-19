## Numpy初识

#### 核心数据结构

- numpy中的核心数据结构是ndarray
- 使用type函数可以查看数据类型
- arange()函数参考2.2
```
import numpy as np
txtdata = np.arange(1)
print type(txtdata) # ==> <type 'numpy.ndarray'>
```

#### 使用帮助文档

- 使用help函数可以在命令行打印相关函数的API及示例
- 之后提到的函数记录了常用用法，可以通过help查看API中的详细用法及说明
```
import numpy as np
print help(np.arange)
```
