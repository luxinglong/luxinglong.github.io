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
定理有了，怎么理解呢？
我们知道，有了策略网络就要进行误差的反向传递，以便进行参数的更新。可是对于PG来讲，只有一段发生过的经历，并不能计算误差。那怎么来更新参数呢？这时候奖励就出场了，在那段经历中，每个动作执行之后，都会出现一个即时奖励。那么就可以根据奖励的情况来更新参数。奖励大的，参数更新就使这个动作更大概率出现，而奖励小的，则会减小出现的概率。

策略梯度公式：
$$
\Delta_{\theta}J(\theta)=\mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}log\pi_{\theta}(s,a)Q^{\pi_{\theta}}(s,a)]
$$
将求期望的部分可以分成两个子式：
* 第一部分$\nabla_{\theta}log\pi_{\theta}(s,a)$,这是一个方向向量，代表了$log\pi_{\theta}(s,a)$对于参数$\theta$变化最快的方向，参数在这个方向上更新可以增大或者减小$log\pi_{\theta}(s,a)$，也就是增大或者减小$(s,a)$的概率。

* 第二部分$Q^{\pi_{\theta}}(s,a)$,状态价值函数，这是一个标量，在策略梯度中扮演着在$\nabla_{\theta}log\pi_{\theta}(s,a)$方向上变化幅度的角色。如果$Q^{\pi_{\theta}}(s,a)$比较大，$log\pi_{\theta}(s,a)$的参数在这个方向上变化的幅度也比较大，那么$(s,a)$的概率也会比较大。

所以，对于梯度策略最直观的理解就是增大高回报动作-状态的概率，减小低回报动作-状态的概率。

**基于MC的策略梯度**
伪代码为：
{% img [mcpg] http://on99gq8w5.bkt.clouddn.com/mcpg.png?imageMogr2/thumbnail/400x400 %}
怎么理解这段代码呢？
首先，随机初始化策略网络的参数。
然后，要准备好一段经历，以备训练。
接着，对经历中的每一步，根据返回的状态价值，进行参数更新。

有两个疑问，一是经历从哪里来？二是状态价值$v_t$怎么计算？

经历的来源：就是利用初始化的策略网络$\pi_{\theta}(s)$，根据初始状态$s_0$，产生初始动作$a_0$，然后执行动作$a_0$，产生新的状态$s_1$和即时奖励$r_1$，最后将$\langle s_0, a_0, r_1, s_1 \rangle$保存到“记忆”中。如此反复直到遇到终结状态为止，此时便产生了一段经历Episode.

状态价值：$v_0=r_1, v_t=r_{t+1}+\gamma v_{t-1}$ 

# 参考文献
[1] David Silver, reinforcement learning lecture 7
