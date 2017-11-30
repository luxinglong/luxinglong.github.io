---
title: 【强化学习】CADRL仿真算法实现
date: 2017-11-13 01:19:12
tags:
    - RL
    - robotics
    - local planning
categories: 【强化学习】
---
{% img [CADRL] http://on99gq8w5.bkt.clouddn.com/CADRL.png?imageMogr2/thumbnail/500x500 %}
<!--more-->
# 0 引言
CADRL用神经网络来估计CA问题的状态值函数，然后将连续的动作空间离散化成35个动作，由此可以采用的策略就是选择取得最大化即时奖励与下一个状态的价值和的动作。即
$$
a_t = arg\max\limits_{a_t\in A}[R(s_t^{jn}, a_t)+\bar{\gamma}V(\hat{s}_{t+1}, \hat{\tilde{s}}_{t+1}^o)] \\
\bar{\gamma}=\gamma^{\Delta t\cdot v_{pref}} \\
\hat{\tilde{s}}_{t+1}^o \gets propagate(\tilde{s}_t^o, \Delta t\cdot \hat{\tilde{v}}_t) \\
\hat{s}_{t+1} \gets propagate(s_t, \Delta t\cdot a_t)
$$

文章解释，之所以不估计动作-状态值函数$Q(s^{jn},a)$，而是选择估计状态值函数$V(s^{jn})$，是因为两点：
1. CA问题的动作空间是连续的，没有办法直接估计$Q(s^{jn},a)$；
2. 动作空间依赖于agent的状态，比如说航向角影响方向、优先选择的速度值影响大小。

# 1 训练数据准备
预训练，为了获得更好的网络初始参数。
网络结构：
{% img [DVN] http://on99gq8w5.bkt.clouddn.com/DVN.png?imageMogr2/thumbnail/300x300 %}

ORCA根据初始化的位置，产生两条轨迹。多次执行这个程序，可以得到多条轨迹。把这些轨迹拆开，处理成一个“状态-价值”对的集合。
$$
\{s^{jn},y\}^N_{k=1}
$$
其中，$y=\gamma^{t_g\cdot v_{pref}}$,$t_g$表示到达目标的时间。因为$\gamma$小于1，而$t_g\cdot v_{pref}$可以表示从当前位置到达目标的路程，路程越大，价值就会越小。

损失函数：
$$
loss=arg\min\limits_{w}\sum_{k=1}^N(y_k-V(s^{jn}_k;w))^2
$$

难点：怎么生成训练数据？怎么确定初始化位置？怎么计算$t_g$？
ORCA生成的轨迹格式为：$\{t_0, p_0, \cdots, t_T, p_T\}$
那么$t_g=t_T - t_i$, $v_{pref}=0.5$

要生成的数据格式是怎么样的呢？
$time$  $d_g$  $v_{pref}$  $v_x^{\prime}$  $v_y^{\prime}$  $r$  $\theta^{\prime}$  $\tilde{v}_x^{\prime}$  $\tilde{v}_y^{\prime}$  $\tilde{p}_x^{\prime}$  $\tilde{p}_x^{\prime}$  $\tilde{r}$  $r+\tilde{r}$  $cos(\theta^{\prime})$ $sin(\theta^{\prime})$ $d_a$ $y=\gamma^{t_g\cdot v_{pref}}$

机器人初始位置和目标位置的确定，需要满足以下几个条件：
1. 包含尽可能多的情况，包括正常会车、超车、交叉等情况；
2. 确定局部路径规划区域的大小；

现在把生成数据的方式确定为四种情况，交叉、相会、超车、其他。
交叉：

# 2 CADRL生成轨迹
算法伪代码：
{% img [predict_cadrl] http://on99gq8w5.bkt.clouddn.com/predict_cadrl.png?imageMogr2/thumbnail/400x400 %}

# 3 CADRL训练
算法伪代码：
{% img [train_cadrl] http://on99gq8w5.bkt.clouddn.com/train_cadrl.png?imageMogr2/thumbnail/400x400 %}
首先是监督学习的初始化过程。
忽略了最关键的一点，就是怎么制作特征？
在准备数据的过程中，我发现，另外一个机器人的位置是可以随意指定的，而且要把这个机器人的位置的横纵坐标作为特征，那么如果位置选的比较大，那么就会导致位置这个特征，对结果的影响比较大。另外一个问题，如何生成那么多情况的轨迹呢？

一开始，不太理解这里，慢慢理解了。最优策略应该与坐标系无关，即与旋转和平移无关。那么怎么解决这个问题呢？引入一个以机器人为中心的坐标系，原点在机器人中心，x轴指向目标位置。注意：这个坐标系只在生成特征的时候使用。

对神经网络的训练，有几个关键点，一是，训练数据不具有相关性，也就是说要把生成的数据打乱；二是，数据要进行归一化，以防止训练某些元素的比重过大。

# 4 实验结果

# 参考文献
[1] Chen, Yu Fan, et al. "Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning." Robotics and Automation (ICRA), 2017 IEEE International Conference on. IEEE, 2017.