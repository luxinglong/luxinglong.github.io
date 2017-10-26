---
title: 【强化学习】免模型学习(Model-Free)
date: 2017-10-20 18:57:59
tags:
    - RL
    - robotics
categories: 【强化学习】
---

# 0 引言
在现实的强化学习任务中，环境的模型，也就是状态转移概率、奖赏函数不易得知，因此需要进行免模型学习。

<!--more-->

# 1 评估
在没有环境模型的情形下，策略迭代面临两大困难：一是无法对一个给定的策略进行评估，因为P、R未知，无法得到Bellman等式。二是无法建立状态值函数和动作-状态值函数的联系。(策略迭代算法估计的是最优状态值函数，而最优策略要由最优动作-状态值函数获得)
## 1.1 蒙特卡罗
对于问题一，蒙特卡罗采用采样的办法，即通过在环境中执行动作，观察环境的状态转移和得到的奖赏，经过多次“采样”，求取平均累积奖赏作为期望累积奖赏。对于问题二，蒙特卡罗绕过状态值函数，直接估计动作-状态值函数。

采样获得的“轨迹”或者“片段”，是基于一个策略$\pi$生成的。
episode：$\langle x_0,a_0,r_1,x_1,a_1,r_2,\cdots,x_{T-1},a_{T-1},r_T,x_T\rangle \sim \pi$

在计算累积奖励时，如果某个状态在经过几次转换之后，又回到了这个状态，那么这个状态记录几次呢？两种策略，First-Visit只记录第一次出现时的一次，Every-Visit每次出现都会记录。

$$
G_t = r_{t+1}+\gamma r_{t+2}+ \cdots + \gamma^{T-1}r_{T}
$$

为了对策略$\pi$进行评估，需要增量地计算值函数
对于每一个状态$x_t$和累积奖励$G_t$
$$
V(x_t) \gets V(x_t)+\frac{1}{N(x_t)}(G_t-V(x_t)) \\
其中，N(x_t) \gets N(x_t)+1
$$

在某些情况下（不太理解），可以将$\frac{1}{N(x_t)}$换成一个固定大小的值$\alpha$
$$
V(x_t) \gets V(x_t)+\alpha (G_t-V(x_t)) 
$$
## 1.2 时序差分TD
蒙特卡罗方法必须从完成的episode中进行学习效率较低，而时序差分结合了蒙特卡罗和动态规划的思想，可以从不完整的episode中学习。

### 1.2.1 TD(0)
在TD(0)中使用估计的返回值$R_{t+1}+\gamma V(x_{t+1})$，而不是采样计算的$G_t$来更新状态值，
$$
V(x_t) \gets V(x_t)+\alpha (R_{t+1}+\gamma V(x_{t+1}-V(x_t)) 
$$
其中，$R_{t+1}+\gamma V(x_{t+1}$称为TD的目标；
     $\delta_{t}=R_{t+1}+\gamma V(x_{t+1}-V(x_t)$称为TD误差。

**TD和MC的区别一**
* TD可以在得到最后的结果前进行学习：
TD可以每一步后在线学习
MC必须等到整个episode结束后才可以进行学习
* TD可以在没有最后结果的情况下学习：
TD可以从不完整的序列中学习
MC只能从完整的序列中学习
TD可以在连续状态（没有结束状态）的环境中工作
MC只能在包含完整episode的情况下工作

**TD和MC的区别二**
* MC有着比较高的方差variance，没有偏移误差bias
$G_t = r_{t+1}+\gamma r_{t+2}+ \cdots + \gamma^{T-1}r_{T}$是对$v_{\pi}(x_t)$的无偏估计
高方差：返回$G_t$依赖于很多随机的动作、状态转移和奖励
对初值不太敏感
* TD有着比较低的方差variance，有一些偏移误差bias
$R_{t+1}+\gamma V(x_{t+1}$是对$v_{\pi}(x_t)$的有偏估计
低方差：只依赖于一个随机的动作、状态转移和奖励
对初值敏感

**TD和MC的区别三**
* TD利用了马尔可夫特性
* MC没有用到马尔可夫特性

### 1.2.2 TD$(\lambda)$


**效用追踪**(Eligibility Traces)
例子：<铃响、铃响、铃响、灯亮、电击>，请问是铃响还是灯亮导致了电击发生？
$$
E_0(x)=0 \\
E_t(x)=\gamma \lambda E_{t-1}(x)+1(X_t=x)
$$




# 2 控制

## 2.1 蒙特卡罗 

{% img [mc] http://on99gq8w5.bkt.clouddn.com/mc.png?imageMogr2/thumbnail/400x500 %}
这个算法需要记录两个值，一个是每个动作-状态对出现的次数，另外一个是每个状态的动作-状态值函数。
策略评估使用
$$
V(x_t) \gets V(x_t)+\frac{1}{N(x_t)}(G_t-V(x_t)) \\
其中，N(x_t) \gets N(x_t)+1
$$
对于，返回值$G_t$，使用。。。。来计算。

策略提升环节，使用$\epsilon -$贪婪法
$$
\epsilon \gets \frac{1}{s} \\
\pi \gets \epsilon - greedy(Q)
$$
最终，动作-状态值函数会收敛到最优，$Q(x,a) \to q_{\ast}(x,a)$.

**GILE**(Greedy in the Limit with Infinite Exploration),就是在有限的时间进行无限可能的探索。

## 2.2 Sarsa
在评估策略时，TD相比MC有很多优点，比如说小方差、在线学习、根据不完整的episode学习等。同样的在控制问题上，也可以利用TD的这些优点，如Sarsa算法。
为什么取名字叫Sarsa呢？因为算法的求解需要用到上一时刻的状态S，上一时刻的动作A，当前的奖赏R，当前状态$S^{\prime}$，当前动作$A^{\prime}$.
对于状态-动作值函数的估计采用下面的公式：
$$
Q(X,A)\gets Q(X,A)+\alpha (R+\gamma Q(X^{\prime},A^{\prime})-Q(X,A))
$$

{% img [sarsa] http://on99gq8w5.bkt.clouddn.com/sarsa.png?imageMogr2/thumbnail/400x500 %}

Sarsa算法收敛到最优状态-动作值函数的收敛条件：
* 任何时候的策略$\pi_t(a \mid x)$符合GLIE特性；
* 步长系数$\alpha_t$满足：$\sum^{\infty}_{t=1}\alpha_t=\infty$，且$\sum^{\infty}_{t=1}\alpha^2_t < \infty$
看不懂。。

## 2.3 Sarsa$(\lambda)$



## 2.4 Q-learning
Sarsa是同策略算法，也就是说评估和提升的策略是同一个。如果将Sarsa改成异策略，那么就得到类Q-learning算法。这时候评估和提升的策略不是同一个。
{% img [ql] http://on99gq8w5.bkt.clouddn.com/ql.png?imageMogr2/thumbnail/400x500 %}


#  参考文献
[1] 周志华,《机器学习》,清华大学出版社,2016
[2] David Silver, reinforcement learning lecture 4 and 5
