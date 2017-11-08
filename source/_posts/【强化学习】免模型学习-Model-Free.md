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

TD(0)在策略评估时，只是在当前状态往前走一步，如果要是多走几步再更新状态值函数，就会引出n-step预测。
**n-Step Prediction**

{% img [n-step] http://on99gq8w5.bkt.clouddn.com/n-step.png?imageMogr2/thumbnail/400x500 %}

n-step return:    $G_t^{(n)}=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^nV(X_{t+n})$
那么n-step的TD学习模型为：
$$
V(x_t) \gets V(x_t)+\alpha(G_t^{(n)}-V(x_t))
$$
那么在不同的$\alpha$和$n$，效果会怎么样呢？

如何在不增加计算量的前提下，综合考虑不同的步数预测？为此引出，TD$(\lambda)$。
$\lambda$-return:   $G_t^{\gamma}=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}$
为$G_t^{(n)}$增加一个$(1-\lambda)\lambda^{n-1}$的权重，然后累加。权重单调递减，衰减到零。
那么TD$(\lambda)$学习模型为：
$$
V(x_t) \gets V(x_t)+\alpha(G_t^{\lambda}-V(x_t))
$$

由于TD$(\lambda)$的return考虑了所有步长的预测，因此需要完整的episode，这也带来了MC算法的计算效率问题。$\lambda$取值范围是0-1，事实上，当$\lambda=1$，TD$(\lambda)$等效于MC算法。为了解决这个问题，引入效用追踪。

**效用追踪**(Eligibility Traces)
例子：<铃响、铃响、铃响、灯亮、电击>，请问是铃响还是灯亮导致了电击发生？
频率启发(Frequency heuristic)：将原因归于出现频率最高的状态，如铃响
就近启发(Recency heuristic)：将原因归于最近的状态，如灯亮
效用追踪综合考虑频率启发和就近启发。
$$
E_0(x)=0 \\
E_t(x)=\gamma \lambda E_{t-1}(x)+1(X_t=x)
$$
下图给出了$E_t(x)$对于t的一个可能的曲线图：
{% img [et] http://on99gq8w5.bkt.clouddn.com/et.png?imageMogr2/thumbnail/400x500 %}
该图横坐标是时间，横坐标下面的竖线的位置代表当前进入了状态x，纵坐标是效用追踪值$E_t(x)$。可以看出当一个状态连续出现时，E值就会在一定衰减的基础上有一个单位数值的提高，此时将增加该状态对于最终收获贡献的比重。如果该状态距离最终状态比较远，则其对最终收获的贡献越小。

特别地，E值不需要等到完整的episode结束才能计算出来，它可以每经过一个时刻就得到更新。E值存在饱和值，有一个瞬时上限：$E_{\max}=1(1-\gamma\lambda)$

$$
V(x)\gets V(x)+\alpha\delta_t E_t(x) \\
其中，\delta_t=R_{t+1}+\gamma V(x_{t+1})-V(x_t)为TD-error \\
     E_t(x)是效用追踪
$$

注意：每个状态x都有一个E值，E值随时间而变化。

这样就可以在线实时更新状态值函数，而不用使用完整的序列。同时，通过效用追踪，可以将状态出现的频率和就近性考虑进状态值函数。

注：ET是一个非常符合神经科学相关理论的、非常精巧的设计。把它看成是神经元的一个参数，它反映了神经元对某一刺激的敏感性和适应性。神经元在接受刺激时会有反馈，在持续刺激时反馈一般也比较强，当间歇一段时间不刺激时，神经元又逐渐趋于静息状态；同时不论如何增加刺激的频率，神经元有一个最大饱和反馈。

# 2 控制

## 2.1 蒙特卡罗 

{% img [mc-on] http://on99gq8w5.bkt.clouddn.com/mc-on.png?imageMogr2/thumbnail/400x500 %}
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
Sarsa算法的实现有两种思路，一是值迭代[1],二是策略迭代[2].值迭代如下图所示：
{% img [sarsa1] http://on99gq8w5.bkt.clouddn.com/sarsa1.png?imageMogr2/thumbnail/500x500 %}

策略迭代如下图所示：
{% img [sarsa2] http://on99gq8w5.bkt.clouddn.com/sarsa2.png?imageMogr2/thumbnail/400x400 %}

Sarsa算法收敛到最优状态-动作值函数的收敛条件：
* 任何时候的策略$\pi_t(a \mid x)$符合GLIE特性；
* 步长系数$\alpha_t$满足：$\sum^{\infty}_{t=1}\alpha_t=\infty$，且$\sum^{\infty}_{t=1}\alpha^2_t < \infty$
看不懂。。

## 2.3 Sarsa$(\lambda)$
**n-Step Sarsa**
n-step Q-return:   $q_t^{(n)}=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^nQ(x_{t+n})$

这里的$q_t$对应的是一个状态行为对$\langle x_t,a_t\rangle$，表示在某个状态下执行某个动作的价值大小。

对于$q_t^1$,表示状态行为对$\langle x_t,a_t\rangle$的Q价值可以分成两部分。一部分是执行动作$a_t$离开状态$x_t$的即时奖励$R_{t+1}$，即时奖励只与状态有关，与该状态下采取的行为无关；另一部分是新状态行为对$\langle x_{t+1},a_{t+1}\rangle$的Q价值：环境给了个体一个新状态$x_{t+1}$，观察在$x_{t+1}$状态时基于**当前策略**得到的行为$a_{t+1}$时的$Q(x_{t+1},a_{t+1})$，后续的Q价值考虑衰减系数。

n-Step Sarsa学习模型为：
$$
Q(x_t,a_t)\gets Q(x_t,a_t)+\alpha(q_t^{(n)}-Q(x_t,a_t))
$$

**Sarsa$(\lambda)$**
如果给每一个n-step Q-return分配一个权重，并求和，就会得到$q^{\lambda}$-return，它结合了所有n-step Q-return：
$$
q_t^{\lambda}=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}q_t^{(n)}
$$

**Sarsa$(\lambda)$前向认识**
使用$q_t^{\lambda}$来替换状态-动作价值更新递推公式中的返回，那么可以得到Sarsa$(\lambda)$学习模型为：
$$
Q(x_t,a_t)\gets Q(x_t,a_t)+\alpha(q_t^{\lambda}-Q(x_t,a_t))
$$
这就是前向认识的Sarsa$(\lambda)$，它需要完整的episode才能完成更新。

**Sarsa$(\lambda)$反向认识**
与TD$(\lambda)$类似，这里同样引入效用追踪来达到在线更新的效果。不过这次E值针对的不是一个状态，而是一个动作-状态对：
$$
E_0(x,a)=0 \\
E_t(x,a)=\gamma \lambda E_{t-1},a(x)+1(X_t=x,A_t=a)
$$

综合考虑TD-error和效用追踪可以得到新的学习模型：
$$
Q(x_t,a_t)\gets Q(x_t,a_t)+\alpha\delta_tE_t(x,a) \\
其中，\delta_t=R_{t+1}+\gamma Q(x_{t+1},a_{t+1})-Q(x_t,a_t)为TD-error \\
     E_t(x,a)是效用追踪
$$

Sarsa$(\lambda)$的伪代码如下：
{% img [sarsa-l] http://on99gq8w5.bkt.clouddn.com/sarsa-l.png?imageMogr2/thumbnail/400x500 %}

这里要提及一下的是E(s,a)在每遍历完一个Episode后需要重新置0，这体现了ET仅在一个Episode中发挥作用；其次要提及的是算法更新Q和E的时候针对的不是某个Episode里的Q或E，而是针对个体掌握的整个状态空间和行为空间产生的Q和E。

## 2.4 Q-learning
Sarsa是同策略算法，也就是说评估和提升的策略是同一个。如果将Sarsa改成异策略，那么就得到类Q-learning算法。这时候评估和提升的策略不是同一个。
{% img [qlearning] http://on99gq8w5.bkt.clouddn.com/qlearning.png?imageMogr2/thumbnail/400x400 %}

或者可以这样理解，异策略就是说产生学习样本的策略和实际提升的策略不是同一个。如伪代码中，产生样本的策略是预先定义好的随机策略$\pi(x,a)$，而实际提升的是贪婪策略。对于Sarsa来说，没有提前定义好的策略，只是会初始化一个策略，然后用它来产生样本，然后再对其进行提升。是同一个策略。

有了异策略，那么agent就不需要自己亲力亲为，自己从初始策略慢慢迭代了。而可以从别人学习的经历中进行学习。看别人学习而学习。


#  参考文献
[1] 周志华,《机器学习》,清华大学出版社,2016
[2] David Silver, reinforcement learning lecture 4 and 5
[3] 叶强, David Silver强化学习公开课中文讲解及实践, 知乎专栏
