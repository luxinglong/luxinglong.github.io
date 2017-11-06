---
title: 【强化学习】策略梯度-Policy Gradient
date: 2017-11-02 13:00:04
tags:
    - RL
    - robotics
categories: 【强化学习】
---
{% img [pg] http://on99gq8w5.bkt.clouddn.com/pg.png?imageMogr2/thumbnail/600x600 %}
<!--more-->
# 0 引言
之前一讲，通过值函数逼近，获得状态值函数或者动作-状态值函数的估计值，然后采用贪婪法获得最优策略。本讲中，直接用函数逼近策略，即
$$
\pi_{\theta}(s,a)=\mathbb{P}[a|s,\theta]
$$

这样做的优点是：
* 更好的收敛特性
* 在高维或者连续动作空间中更加有效
* 可以学习到随机策略

当然，也有缺点：
* 通常收敛到一个局部最优解而不是全局最优解
* 策略评估通常不够高效且具有很大的方差

# 1 策略梯度
**状态重名**

**策略目标函数**
策略梯度法的目标是：给定一个带有参数$\theta$的策略$\pi_{\theta}(s,a)$,寻找最好的参数$\theta$
三类目标函数：
在有始有终的环境中，目标函数为开始状态值函数的期望：
$$
J_1(\theta)=V^{\pi_{\theta}}(s_1)=\mathbb{E}_{\pi_{\theta}}[v_1]
$$
在连续的环境中，利用状态分布求得所有状态值函数的期望：
$$
J_{av}(\theta)=\sum_s d^{\pi_{\theta}}(s)V^{\pi_{\theta}}(s)
$$
平均每步回报
$$
J_{avR}(\theta)=\sum_s d^{\pi_{\theta}}(s)\sum_a\pi_{\theta}(s,a)\mathcal{R}^a_s
$$
其中，$d^{\pi_{\theta}}(s)$是一个稳定的状态分布。

**策略优化**
基于策略的强化学习就是一个最优化问题，目标是寻找一个参数$\theta$最大化目标函数。
求解最优化问题，可以使用梯度方法，也可以使用其他方法。在这里，我们使用基于梯度的方法。

定义策略梯度：
$$
\Delta_{\theta}J(\theta)=(\frac{\partial J(\theta)}{\partial \theta_1},\cdots,\frac{\partial J(\theta)}{\partial \theta_n})^T
$$
参数更新：
$$
\Delta \theta=\alpha \nabla_{\theta}J(\theta)
$$

通过有限差分的方式来计算策略梯度：
$$
\frac{\partial J(\theta)}{\partial \theta_k} \approx \frac{J(\theta+\epsilon u_k)-J(\theta)}{\epsilon}
$$
其中，$u_k$是一个单位向量。第$k$个元素为1，其他元素为0

得分函数Score Function
$$
\nabla_{\theta}\pi_{\theta}(s,a)=\pi_{\theta}(s,a)\frac{\nabla_{\theta}\pi_{\theta}(s,a)}{\pi_{\theta}(s,a)}=\pi_{\theta}(s,a)\nabla_{\theta}log\pi_{\theta}(s,a)
$$
其中，$\nabla_{\theta}log\pi_{\theta}(s,a)$为得分函数。
Softmax策略
对于离散的动作空间，可以用Softmax策略来逼近真实的策略，它的输出为采取各个动作的概率。即
$$
\pi_{\theta}(s,a)\propto e^\phi(s,a)^T\theta
$$
这样，得分函数为：
$$
\nabla_{\theta}log\pi_{\theta}(s,a)=\phi(s,a)-\mathbb{E}_{\pi_{\theta}}[\phi(s,\cdot)]
$$
Gaussian策略
当动作空间是连续的时候，采用高斯策略更为合理，输出采取各个动作的分布。即
$$
a \sim \mathcal{N}(\eta(s),\sigma^2)
$$
这样，得分函数为：
$$
\nabla_{\theta}log\pi_{\theta}(s,a)=\frac{(a-\eta(s))\phi(s)}{\sigma^2}
$$

**策略梯度定理**
{% img [pg_t] http://on99gq8w5.bkt.clouddn.com/pg_t.png?imageMogr2/thumbnail/500x500 %}

**基于MC的策略梯度**
伪代码为：
{% img [mcpg] http://on99gq8w5.bkt.clouddn.com/mcpg.png?imageMogr2/thumbnail/400x400 %}

# 参考文献
[1] David Silver, reinforcement learning lecture 7
