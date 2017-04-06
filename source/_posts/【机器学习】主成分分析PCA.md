---
title: 【机器学习】主成分分析PCA
date: 2017-03-23 11:38:20
tags: 
    - machine leaning
    - pattern recognition
    - deep learning]
categories: 【机器学习】
---

# 0 背景知识
在机器学习中，当测试样本的维数很高时，会出现数据样本稀疏，距离计算困难等问题，这被称为“维数灾难”(curse of dimensionality). 为了解决维数灾难带来的影响，提出了降维(dimension reduction)操作。降维主要分为线性降维和非线性降维。线性降维包括多维缩放MDS和主成分分析PCA。非线性降维有核主成分分析KPCA和流形学习等方法。本文主要介绍PCA.

>其实，现在项目中还没有用到主成分分析PCA，但是看到百度计算机视觉工程师面试中会问到PCA的相关问题，所以总结一下，以备不时之需。

---------------------------------------------------------
<!--more-->

# 1 基本原理
{% blockquote %}
“为什么能进行降维？这是因为在很多时候，人们观测或收集到的数据样本虽然是高维的，但是与学习任务密切相关的也许仅是某个低维分布”[1]。
{% endblockquote %}

{% img [降维] http://on99gq8w5.bkt.clouddn.com/%E9%99%8D%E7%BB%B4.jpg?imageMogr2/thumbnail/300x300 降维示意图 %}

问题描述：

假设有m个样本$\{x^{(1)},\cdots,x^{(m)}\}$，每个样本还有n个属性，即$x^{(i)} \in R^n$.现在如果有一个矩阵D,使得$c^{(i)} = D \ast x^{(i)},c^{(i)} \in R^l$,且$l < n$.

$$
\begin{eqnarray}
\nabla\cdot\vec{E} &=& \frac{\rho}{\epsilon_0} \\
\nabla\cdot\vec{B} &=& 0 \\
\nabla\times\vec{E} &=& -\frac{\partial B}{\partial t} \\
\nabla\times\vec{B} &=& \mu_0\left(\vec{J}+\epsilon_0\frac{\partial E}{\partial t} \right)
\end{eqnarray}
$$

## 1.1 推导方式一

## 1.2 推导方式二

## 1.3 推导方式三

# 2 算法实现
## 2.1 伪码

## 2.2 C++

{% codeblock hello.cpp %}
#include <iostream>
using namespace std;
int main(void)
{
    cout << "hello hexo" << endl;
    return 0;
}
{% endcodeblock %}

# 3 结论

# 参考文献
[1] 周志华. 机器学习. 清华大学出版社. (2016)
[2] 

