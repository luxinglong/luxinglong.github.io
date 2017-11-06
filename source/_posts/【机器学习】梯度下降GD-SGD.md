---
title: 【机器学习】梯度下降GD,SGD
date: 2017-11-06 10:20:17
tags:
    - machine learning
categories: 【机器学习】
---
# 0 引言
对于优化问题，经常使用的就是梯度下降算法，但是始终没有搞清楚GD和SGD的差别，翻看了《深度学习》找到了差别，特此记录。

问题描述：对每个样本的损失函数求期望
$$
J(\theta) = \mathbb{E}_{x,y \sim \hat{p}_{data}}L(x,y,\theta)=\frac{1}{m}\sum^{m}_{i=1}L(x^{(i)},y^{(i)},\theta)
$$
<!--more-->
# 1 GD
$$
\nabla_{\theta}J(\theta)=\frac{1}{m}\sum^{m}_{i=1}\nabla_{\theta}L(x^{(i)},y^{(i)},\theta)
$$
由于梯度下降要对所有的样本的损失函数求期望，所以当样本容量很大的时候，这种方法就不太实用了。但是该方法可以收敛到全局最优解。
# 2 SGD
随机梯度下降的想法就是用样本集合一个比较小的集合来估计这个期望。通常子集合的容量在1到几百之间。
$$
g=\frac{1}{m^{\prime}}\sum^{m^{\prime}}_{i=1}\nabla_{\theta}L(x^{(i)},y^{(i)},\theta)
$$
然后使用随机梯度来更新参数：
$$
\theta \gets \theta - \eta g
$$
其中，$\eta$是学习率。

缺点是SGD的噪音较GD要多，使得SGD并不是每次迭代都向着整体最优化方向。虽然SGD的训练速度快，但是准确率下降了，也不能得到全局最优解。

# 参考文献
[1] DEEP LEARNING, Yoshua Bengio
