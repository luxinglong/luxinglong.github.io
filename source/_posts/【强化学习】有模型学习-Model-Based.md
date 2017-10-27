---
title: 【强化学习】有模型学习(Model-Based)
date: 2017-10-20 18:57:34
tags:
    - RL
    - robotics
categories: 【强化学习】
---

{% img [iteration] http://on99gq8w5.bkt.clouddn.com/policy_iteration.png?imageMogr2/thumbnail/500x400 %}

<!--more-->

# 0 引言

**有模型学习**，通常的说法是智能体(agent)对环境进行了建模，可以不观察环境，就能模拟出与环境相同或近似的情况。更容易理解的说法是，对任意状态$x, x^{\prime}$和动作$a$，在$x$状态下执行动作$a$转移到$x^{\prime}$的概率$P^{a}_{x\to x^{\prime}}$是已知的，该转移带来的奖赏$R^{a}_{x\to x^{\prime}}$也是已知的。

举个例子：
{% img [student] http://on99gq8w5.bkt.clouddn.com/student.png?imageMogr2/thumbnail/500x400 %}

在有模型的情况下，下面将要讨论，如何对一个策略的好坏进行评估，那么就会引出值函数和Bellman方程；如果一个策略不是很好，那么如何进行改进？策略迭代和值迭代就会出现。

**马尔可夫决策过程**，$MDP=\langle X,A,P,R,\gamma \rangle$

# 1 策略评价
如果现在有个策略$\pi$，那么如何评价该策略是好是坏呢？一般是估计该策略带来的期望累积奖赏。也就是值函数，下面定义值函数，值函数有两种定义方式：

T步累积奖励
$$
V^{\pi}_{T}(x)=\mathbb{E}_{\pi}[\frac{1}{T}\sum_{t=1}^{T}r_{t}\mid x_0=x]
$$
$\gamma$折扣累积奖励
$$
V^{\pi}_{\gamma}(x)=\mathbb{E}_{\pi}[\sum_{t=0}^{+\infty}\gamma^{t}r_{t+1}\mid x_0=x]
$$

但是，如果写成这样，给定一个MDP和策略$\pi$，怎么去计算相应的值函数呢？显然，不太容易。所以这里采用了递归的形式，把这个问题分解成小问题。由此便可以得到**Bellman等式**。
$$
V^{\pi}_{T}(x)=\sum_{a\in A}\pi(x,a)\sum_{x^{\prime}\in X}P^{a}_{x\to x^{\prime}}(\frac{1}{T}R^{a}_{x\to x^{\prime}}+\frac{T-1}{T}V^{\pi}_{T-1}(x^{\prime}))
$$
$$
V^{\pi}_{\gamma}(x)=\sum_{a\in A}\pi(x,a)\sum_{x^{\prime}\in X}P^{a}_{x\to x^{\prime}}(R^{a}_{x\to x^{\prime}}+\gamma V^{\pi}_{\gamma}(x^{\prime}))
$$

注意：一个策略对应一个值函数，

# 2 策略改进
如何判断策略改进呢？那就是改进后的策略对应的值函数优于改进前的策略。

$$
\pi^{\prime}(x)=arg\max\limits_{a\in A} q_{\pi}(x,a)
$$
就是说，在现有的策略基础上，对每一个状态，求使该状态下动作-状态值函数获得最大的动作，作为新的策略。这个方法也被成为贪婪法。
那么怎么证明$\pi^{\prime}$优于$\pi$呢？
$$
q_{\pi}(x,\pi^{\prime}(x))=\max\limits_{a\in A}q_{\pi}(x,a)\ge q_{\pi}(x,\pi(x))=V_{\pi}(x)
$$
所以说，策略改进后对应的值函数优于改进前的值函数，$V_{\pi^{\prime}}(x)\ge V_{\pi}$

贪婪法：是一种不追求最优解，只希望得到较为满意的解的方法。因为它省去了为找最优解而穷尽所有可能所需的时间，因而可以快速找到满意的解。贪婪法在求解的过程中，每一步都选取一个局部最优的策略，把问题规模缩小，最后把每一步的结果合并起来，形成一个全局解。

# 3 策略迭代和值迭代
如文章开头的图示，策略迭代就是从一个初始的策略开始，不断进行策略评估和策略改进，直到收敛到最优的策略为止。伪代码如下：
{% img [pi] http://on99gq8w5.bkt.clouddn.com/pi.png?imageMogr2/thumbnail/500x400 %}

策略迭代和值迭代是等价的：
证明：

值迭代就是从一个初始化的状态值函数开始，不断进行策略评估，并按照Bellman等式更新状态值函数，直到状态值收敛为止。伪代码如下：
{% img [vi] http://on99gq8w5.bkt.clouddn.com/vi.png?imageMogr2/thumbnail/500x400 %}

具体实现见：【强化学习】算法实践-Small GridWorld

# 参考文献
[1] 周志华,《机器学习》,清华大学出版社,2016
[2] David Silver, reinforcement learning lecture 2 and 3
