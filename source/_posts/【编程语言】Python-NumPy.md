---
title: 【编程语言】Python-NumPy
date: 2017-11-13 19:32:02
tags:
    - Python
    - NumPy
categories: 【编程语言】
---

# 0 引言
NumPy是Numerical Python的简称，它深深吸引着使用者的原因有下面几点：
* 底层使用C/C++实现，计算快速且省空间，包含C/C++/Fortran接口；
* 可以进行矢量运算并具有复杂广播能力
* 对整组数据进行快速的标准数学运算，而不需要编写循环
* 具有读写磁盘的数学工具和操作内存映射文件的工具
* 具有线性代数、随机数生成、傅立叶变换等功能
<!--more-->

没有系统学习过NumPy，对NumPy的整体架构不熟悉，所以每次用到的时候都要去翻书，感觉很不方便。所以记下这篇博客，希望可以达到同样的事情只做一遍的效果。

# 1 NumPy组织架构
{% img [numpy] http://on99gq8w5.bkt.clouddn.com/numpy.png?imageMogr2/thumbnail/900x900 %}

# 2 NumPy使用时注意点

## 1 访问数组中的行和列
```Python
array = np.arange(12).reshape((3,4))

for row in array:       # 访问每一行
    print row

for column in array.T:  # 访问每一列
    print column

for item in array.flat: # 访问每一个元素，flat是一个迭代器
    print item
```
## 2 C/C++接口


# 参考文献
[1] 利用Python进行数据分析