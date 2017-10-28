---
title: 【强化学习】算法实践-GridWorld Sarsa Sarsa($\lambda$)
date: 2017-10-28 15:52:36
tags:
    - RL
    - robotics
    - python
categories: 【强化学习】
---
{% img [GW] http://on99gq8w5.bkt.clouddn.com/GW.jpg?imageMogr2/thumbnail/600x600 %}
<!--more-->

# 0 引言
David Silver课程中使用了很多GridWorld的例子，如Samll GridWorld, Random Walk, Windy GridWorld, GridWorld, Cliff Walking. 这些例子中状态空间和动作空间都是离散的有限值，可以用gym中Discrete类来描述，另外这些例子都是用格子来表示世界，动作都是上下左右，所以可以考虑建立一个通用的GridWorld环境类，通过配置格子的数量、属性等来具体描述每一个例子。

# 1 通用的GridWorld环境类


# 2 Sarsa实现

# 3 Sarsa($\lambda$)实现

# 4 总结

# 参考文献
[1] David Silver, reinforcement learning lecture
[2] 叶强, David Silver强化学习公开课中文讲解及实践, 知乎专栏
