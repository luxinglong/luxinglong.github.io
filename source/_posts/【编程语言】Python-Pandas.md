---
title: 【编程语言】Python-Pandas
date: 2017-11-29 19:40:41
tags:
    - Python
    - pandas
categories: 【编程语言】
---
# 0 引言
什么？连pandas都没有用过。请不要告诉我你是搞机器学习的！

```Python
from pandas import Series, DataFrame # Series, DataFrame 使用情况最多，可以引入本地命名空间
import pandas as pd
```
从上面的代码开始，快点开始学习使用吧！
<!--more-->
pandas是基于NumPy构建的，让以NumPy为中心的应用变得更加简单。
# 1 pandas思维导图 
{% img [pandas] http://on99gq8w5.bkt.clouddn.com/pandas.png?imageMogr2/thumbnail/600x600 %}
# 2 pandas使用时的注意点

## 2.1 DataFrame的构造函数
由时间序列、数组、列表、元组组成的字典：
```Python
df2 = pd.DataFrame({'A' : 1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo'})
                    
print(df2)

"""
     A          B    C  D      E    F
0  1.0 2013-01-02  1.0  3   test  foo
1  1.0 2013-01-02  1.0  3  train  foo
2  1.0 2013-01-02  1.0  3   test  foo
3  1.0 2013-01-02  1.0  3  train  foo
"""
```
二维ndarray,index和columns可配置
```Python
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])

"""
             A   B   C   D
2013-01-01   0   1   2   3
2013-01-02   4   5   6   7
2013-01-03   8   9  10  11
2013-01-04  12  13  14  15
2013-01-05  16  17  18  19
2013-01-06  20  21  22  23
"""
```

## 2.2 数据的导入与导出
官方文档：http://pandas.pydata.org/pandas-docs/stable/io.html

# 参考文献
[1] 利用Python进行数据分析
[2] Morvan NumPy & pandas lecture
