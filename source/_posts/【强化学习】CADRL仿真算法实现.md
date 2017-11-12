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

# 1 训练数据准备
预训练，为了获得更好的网络初始参数。

# 2 CADRL生成轨迹
算法伪代码：
{% img [predict_cadrl] http://on99gq8w5.bkt.clouddn.com/predict_cadrl.png?imageMogr2/thumbnail/300x300 %}

# 3 CADRL训练
算法伪代码：
{% img [train_cadrl] http://on99gq8w5.bkt.clouddn.com/train_cadrl.png?imageMogr2/thumbnail/300x300 %}
# 4 实验结果

# 参考文献
[1] Chen, Yu Fan, et al. "Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning." Robotics and Automation (ICRA), 2017 IEEE International Conference on. IEEE, 2017.