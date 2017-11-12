---
title: 【强化学习】CADRL仿真环境搭建
date: 2017-11-08 22:08:11
tags:
    - RL
    - robotics
    - local planning
categories: 【强化学习】
---

{% img [CADRL] http://on99gq8w5.bkt.clouddn.com/CADRL.png?imageMogr2/thumbnail/500x500 %}
<!--more-->
# 0 引言

本文目标就是设计CADRLEnv类，CADRLEnv环境应该可以扩展，从一个Agent扩展到多个Agent，所以应该建立一个Agent类。

# 1 物理引擎
所谓物理引擎就是客观环境的状态转换方程和奖励函数，这里的状态指的是环境的状态，对于部分可观的系统来说，环境的状态和agent的观测不同。
在CADRL的环境中，包括$n$个agent的状态。即
$$
s^i_t=[s_t^{o^i},s_t^{h^i}]=[p_x^i,p_y^i,v_x^i,v_y^i,r^i,p_{gx}^i,p_{gy}^i,v_{pref}^i,\theta^i]\in\Re^9, i=1,2,\cdots n
$$

## 动作空间和观测空间
动作空间：
$$
a(s)=[v_s,\phi], v_s<v_{pref}, \mid \phi - \theta \mid<\pi/6  \\
\mid \theta_{t+1} - \theta_t \mid<\Delta t\cdot v_{pref}
$$

观测空间，agent只能观测到自己的$s_t$和其他agents的$s_t^{o^i}$部分。

## 建立状态转移方程
$$
s^i_{t+1} \gets T*s^i_t
$$
位置：$p_{t+1}=p_t+\Delta t v_t$
速度：$v_{t+1} = a(s_t)$
参考速度：$\theta_{t+1} = tan^{-1}(\frac{v_{t+1,x}}{v_{t+1,y}})$
半径、参考速度、目的地不变。
## 建立奖励函数
$$
\[
R(s^{jn},a)=
\begin{cases}
-0.25& \text{if} \qquad d_{min} < 0 \\
-0.1-d_{min}/2& \text{else if} \qquad d_{min}<0.2 \\
1& \text{else if} \qquad p=p_g \\
0& \text{o.w.}
\end{cases}
\]
$$

编写重要的_reset()函数和_step()函数。
```Python
def _reset(self):
    pass

def _step(self, action):
    pass
```
# 2 图像引擎
需要画出的内容：
agents:圆圈表示，本体agent为红色，其他为非红色
time: text表示
velocity: 箭头表示
trojectory: 线段表示
goal: 实心圆表示，与对应的agent同色

编写重要的_render()函数。
```Python
def _render():
    pass
```
# 3 辅助函数


# 参考文献
[1] Chen, Yu Fan, et al. "Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning." Robotics and Automation (ICRA), 2017 IEEE International Conference on. IEEE, 2017.