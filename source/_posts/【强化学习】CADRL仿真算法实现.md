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

# 2 CADRL生成轨迹
算法伪代码：
{% img [predict_cadrl] http://on99gq8w5.bkt.clouddn.com/predict_cadrl.png?imageMogr2/thumbnail/400x400 %}

# 3 CADRL训练
算法伪代码：
{% img [train_cadrl] http://on99gq8w5.bkt.clouddn.com/train_cadrl.png?imageMogr2/thumbnail/400x400 %}
# 4 实验结果

# 参考文献
[1] Chen, Yu Fan, et al. "Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning." Robotics and Automation (ICRA), 2017 IEEE International Conference on. IEEE, 2017.