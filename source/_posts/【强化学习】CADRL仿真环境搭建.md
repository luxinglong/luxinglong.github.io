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
航向角：$\theta_{t+1} = tan^{-1}(\frac{v_{t+1,y}}{v_{t+1,x}})$
       对于其他agent来说，此信息不可观
优先的、首选的速度：跟全局路径规划的速度保持一致
                 但是，对于局部规划问题来讲，只需要一直指向目标就可以
                 此信息用于约束动作空间
                 在这里，$v_{pref}$指的是速度值，是标量，大小不变
半径、目的地不变。

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

难点一：如何理解并计算$d_{min}$?
定义：按照$t+1$时刻的速度运动，在$\Delta t$内，两个agent最小距离。
计算：解析法
$$
d_{min}=\min \limits_{t} \Arrowvert \vec{p_{t}} - \vec{\tilde{p_t}} \Arrowvert, \qquad t \in [0,\Delta t]
$$
难点二：如何定义到达目的地？
如果严格使当前位置与目标位置重合，可能会导致重合点附近振荡。那么就可以使用一个范围来定义，根据定位精度来确定阈值。

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
goal: 正五角星，重点在给定五角星的外接圆(圆心和半径)，如何求解10个顶点的坐标？
{% img [star] http://on99gq8w5.bkt.clouddn.com/star.jpg?imageMogr2/thumbnail/400x400 %}
假设外接圆半径为$r=10$，圆心为$(0,0)$，则
$A(0,10),\qquad B(-10cos(18^o),10sin(18^o))$
$E(10cos(18^o),10sin(18^o)), \qquad H(10sin(18^o)tan(36^o),10sin(18^o))$
$I(-10sin(18^o)tan(36^o),10sin(18^o)),\qquad F(0,-\sqrt{x_H^2+y_H^2})$
$C(-10sin(36^o), -10cos(36^o)),\qquad D(10sin(36^o), -10cos(36^o))$
$G(\mid y_F \mid cos(18^o),-\mid y_F \mid sin(18^o)),\qquad J(-\mid y_F \mid cos(18^o),-\mid y_F \mid sin(18^o))$

编写重要的_render()函数。
```Python
def _render():
    pass
```
# 3 结果
<video src='http://on99gq8w5.bkt.clouddn.com/orca_demo.mp4' type='video/mp4' controls='controls'  width='100%' height='100%'>
</video>


# 参考文献
[1] Chen, Yu Fan, et al. "Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning." Robotics and Automation (ICRA), 2017 IEEE International Conference on. IEEE, 2017.
[2] https://zhidao.baidu.com/question/2073567152212492428.html