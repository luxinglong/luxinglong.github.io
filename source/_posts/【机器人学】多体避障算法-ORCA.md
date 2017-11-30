---
title: 【机器人学】多体避障算法-ORCA
date: 2017-11-22 13:53:23
tags:
    - robotics
    - path planning
    - collision avoidance
categories: 【机器人学】
---
{% img [ORCA] http://on99gq8w5.bkt.clouddn.com/ORCA.png?imageMogr2/thumbnail/600x600 %}
<!--more-->
# 0 引言
多体避障是机器人学中的一个基础问题。它可以应用在很多领域，比如多体协同控制、群体仿真、无人仓储机器人、AI游戏、机器人在行人环境中自主导航等。多体避障问题也可以描述为**自主导航机器人在一个包含静止和移动障碍物的环境中，规划一条高效、无碰撞、自然的道路。**难点在于需要检测和跟踪移动障碍物，并预测交互后移动障碍物的运动。

主流的研究方法可以分为两大类，集中式和分布式。

集中式的方法依赖于一个可靠的通信网络和中央控制器，每个机器人通过通信网络发布自己的状态并接受行动指令，而中央控制器负责求解一个有约束的最优化问题。这种方法的缺陷在于过于依赖通信，一旦通信出现故障或者延迟比较大，就可能发生碰撞，甚至在行人环境中无法建立通信网络；另外，这种方法难以扩展规模，因为算法的复杂度与机器人的数量呈现高次方相关，计算量太大以至于无法计算或者计算做不到实时。

而分布式的方法可以有效克服上述集中式方法所带来的问题，因此备受青睐。分布式方法可以分为两大类，基于反应的和基于轨迹的。基于反应的方法

评价指标是表现和性能。表现包括效果和效率，效果为各种实验场景下能否完成避障，效率为避障的路径是否最优。性能就是计算的实时性。

本文介绍的ORCA算法是做的比较好的分布式多体避障算法。

# 1 基本概念介绍
**Velocity Obstacles(VO)**
这个概念为自主移动机器人避免与一个已知状态(位置、速度、形态)机器人相撞所需要采取的动作(速度)，提供了充分必要条件。

{% img [VO] http://on99gq8w5.bkt.clouddn.com/VO.png?imageMogr2/thumbnail/600x600 %}

如上图(a)所示，有两个机器人A和B，那么VO被定为$VO^{\tau}_{A\mid B}$，这是一个**相对速度**的集合，如果A机器人采用这个集合中的相对速度，那么就会在$\tau$时间内与B机器人相撞。正式的定义如下：
首先定义圆盘：
$$
D(p,r)=\{q\vert \Vert q-p \Vert < r\}
$$
那么：
$$
VO^{\tau}_{A\vert B}=\{v \vert \exists t \in [0, t] :: tv \in D(p_B-p_A, r_A+r_B)\}
$$
$VO^{\tau}_{A\mid B}$展示在图(b)中，其中$VO^{\tau}_{B\mid A}$关于原点对称。

假设两个机器人当前的速度是$v_A$和$v_B$，如果满足$v_A-v_B \in VO^{\tau}_{A\mid B}$，那么两个机器人就会在时间$\tau$内相撞。

闵可夫斯基和：$X\oplus Y=\{x+y \vert x \in X, y \in Y\}$

如图(c)所示，对于任意一个集合$V_B$，如果满足$v_B \in V_B$，且$v_A \notin VO^{\tau}_{A\mid B} \oplus V_B$，那么A、B机器人至少在时间$\tau$内不会相撞。这就引出了避障速度集合：
$$
CA^{\tau}_{A\vert B}(V_B)=\{v \vert v \notin VO^{\tau}_{A\mid B} \oplus V_B\}
$$

**Reciprocally Collision Avoiding(RCA)**定义为：$V_A \subseteq CA^{\tau}_{A\vert B}(V_B)$，且$V_B \subseteq CA^{\tau}_{B\vert A}(V_A)$
如果满足，$V_A = CA^{\tau}_{A\vert B}(V_B)$，且$V_B = CA^{\tau}_{B\vert A}(V_A)$，那么称之为相互最大(reciprocally maximal)

**Reciprocal Velocity Obstacles(RVO)**
RVO是为了解决机器人和机器人之间避障的问题，而对VO进行的扩展。假设两个机器人都采取避障速度集合中的速度，那么不一定可以在所有情况下完成避障。

**Optimal Reciprocal Collision Avoidance(ORCA)**
ORCA和RCA相比，多了一个最优，就是说，满足$V_A = CA^{\tau}_{A\vert B}(V_B)$，且$V_B = CA^{\tau}_{B\vert A}(V_A)$的条件的$v_A$和$v_B$有很多，但是怎么一个最好的呢？这就是本文的核心模型。

为什么叫相互避障呢？因为现在所处的环境不再是只有静止障碍物，还包含了可以自主避障的其他机器人。其他机器人会根据你的动作

# 2 ORCA定义
{% img [ORCA] http://on99gq8w5.bkt.clouddn.com/ORCA.png?imageMogr2/thumbnail/600x600 %}
如上图所示，ORCA可以定义为：
$$
ORCA^{\tau}_{A \vert B}=\{v \vert (v-(v_A^{opt}+\frac{1}{2}u))\cdot n \leq 0\}
$$
其中，$u$表示$v_A^{opt}-v_B^{opt}$到距离$VO^{\tau}_{A\vert B}$边界最近点的向量。定义如下：
$$
u=(arg \min \limits_{v\in \partial VO^{\tau}_{A\vert B}} \Vert v - (v_A^{opt}-v_B^{opt}) \Vert)-(v_A^{opt}-v_B^{opt})
$$
实质上，$u$就是为了避免A、B机器人相撞，相对速度所需要做出的最小改变。ORCA采用了“责任分担”的机制，就是A、B机器人各自改变$\frac{1}{2}u$.
$n$表示$(v_A^{opt}-v_B^{opt})+u$的外法线。

ORCA标准：RCA且maximal
{% img [ORCA] http://on99gq8w5.bkt.clouddn.com/orca_criterion.png?imageMogr2/thumbnail/600x600 %}
不太理解这个部分。

那么机器人A通过里程计获得自身的位置、速度、半径，并通过传感器观测获得机器人B的位置、速度、半径，就可以得到$ORCA^{\tau}_{A \vert B}$。B机器人同理。
# 3 ORCA算法求解多体避障模型
前面建立了ORCA的避障原则，那么怎么应用于多体避障呢？
{% img [ORCA] http://on99gq8w5.bkt.clouddn.com/orca-multi.png?imageMogr2/thumbnail/600x600 %}
当有多个机器人时，机器人A的可取速度集合表示为$ORCA^{\tau}_A$，其中包含自身的最大速度约束和分别与其他机器人的相对速度约束的交集。
$$
ORCA^{\tau}_A=D(0, v^{max}_A)\cap \bigcap \limits_{B\neq A}ORCA^{\tau}_{A\vert B}
$$
那么接下来，机器人A就可以从$ORCA^{\tau}_A$中选择最靠近优先选择速度$v_A^{pref}$的新速度$v_A^{new}$。
$$
v_A^{new}=arg \min \limits_{v\in ORCA^{\tau}_A}\Vert v - v_A^{pref} \Vert
$$
有了$v_A^{new}$，机器人就可以进行一步的位置更新。
$$
p_A^{new}=p_A+v_A^{new}\Delta t
$$
其中，$\Delta t$是感知-执行循环的时间间隔。

上面最关键的步骤就是根据$ORCA^{\tau}_A$求解$v_A^{new}$。这个问题可以看作有约束的线性规划问题，因为约束组成的集合是一个凸集，有唯一解。通过有效的解法，可以达到$O(n)$的算法复杂度。

一种提高算法效率的办法就是只考虑那些距离A很近的机器人，比如上图中的E机器人，可以不用考虑。文章中使用$kD-tree$来有效地选择添加哪些约束。

关于$kD-tree$，可以参考这篇文章：https://www.cnblogs.com/lysuns/articles/4710712.html

**最优速度$v_A^{opt}$的选择**
{% img [ORCA] http://on99gq8w5.bkt.clouddn.com/orca-empty.png?imageMogr2/thumbnail/600x600 %}
为了通过非通信式的方法解决多体避障的问题，需要每个机器人根据ORCA的原则计算新的速度，但是推理需要每个机器人将自己的最优速度选择暴露给其他机器人。怎么设置这个最优速度呢？
* $v_A^{opt}=0$，将所有机器人的速度设置为0。只考虑其他机器人的位置，不考虑速度。
优点：确保线性规划问题始终优解。
缺点：当机器人很多，比较拥挤的时候，可能造成死锁，所有机器人都不动。如上图(a)所示。
* $v_A^{opt}=v_A^{pref}$，将所有机器人的速度设置为优先选择速度。
优先选择速度是机器人的内部状态，对于其他机器人来说不可观。为了方便讨论，我们假设机器人可以推测其他机器人的优先选择速度。这样选择在“稀疏”环境中，没有问题。当优先选择速度的大小增加时，线性规划可能失效。
* $v_A^{opt}=v_A$，将所有机器人的速度设置为机器人的当前速度。
这种选择是上面两种情况的折中，在“稀疏”环境中倾向于选择最优选择速度，在“稠密”环境中倾向于选择零。而且当前速度可以被其他机器人观测。但是，这样选择也会出现无解的情况，如上图(b)所示，这时候可以利用三维线性规划方法求解“最安全的速度”。

**无解情况的应对**
对于将机器人当前的速度当作所有机器人的最优速度，可能存在无解的情况，如上图(b)所示。这时候我们选择使用三维线性规划求解最安全的速度。
首先，定义$d_{A\vert B}(v)$表示$v$到$ORCA^{\tau}_{A\vert B}$边缘带符号的距离。如果$v\in ORCA^{\tau}_{A\vert B}$，那么$d_{A\vert B}(v)$就是负的。
$$
v_A^{new}=arg \min \limits_{v\in D(0, v_A^{max})} \max \limits_{B\neq A}d_{A\vert B}(v)
$$
求解的公式可以这样理解，
三维线性规划的算法复杂度仍然为$O(n)$.
**对于静止障碍物的处理**
{% img [ORCA] http://on99gq8w5.bkt.clouddn.com/orca-static.png?imageMogr2/thumbnail/600x600 %}
静止障碍物可以表示为线段的集合，因为，静止障碍物没有速度，所以机器人要承担所有的“避障责任”。对于静止障碍物，VO可以定义为：
$$
VO^{\tau}_{A\vert O}=\{v\vert \exists t\in [0,\tau]::tv\in O\oplus - D(p_A, r_A)\}
$$

对于静止障碍物，机器人的最优速度设置为零。

# 4 算法的实现和调参数
调参是机器人工程师的“噩梦”，因为寻找最优的一组参数往往是漫长和乏味的，最重要的是最优参数会随着环境的变化而变化。如何设计一种适应性比较好的算法是科研工作者的追求。

{% img [ORCA] http://on99gq8w5.bkt.clouddn.com/ORCA-params.png?imageMogr2/thumbnail/600x600 %}

代码：
Agent类
{% img [ORCA] http://on99gq8w5.bkt.clouddn.com/orca-agent.png?imageMogr2/thumbnail/600x600 %}

Simulator类
{% img [ORCA] http://on99gq8w5.bkt.clouddn.com/orca-simulator.png?imageMogr2/thumbnail/600x600 %}

Kd-Tree类
{% img [ORCA] http://on99gq8w5.bkt.clouddn.com/orca-kdtree.png?imageMogr2/thumbnail/600x600 %}

# 5 实施
全局路径规划和局部路径规划的结合
{% img [ORCA] http://on99gq8w5.bkt.clouddn.com/orca-sch.png?imageMogr2/thumbnail/600x600 %}

# 6 OCRA仿真结果
<video src='http://on99gq8w5.bkt.clouddn.com/orca_demo.mp4' type='video/mp4' controls='controls'  width='100%' height='100%'>
</video>

# 参考文献
[1]	J. van den Berg, S. J. Guy, M. C. Lin, and D. Manocha, “Reciprocal n-Body Collision Avoidance.,” ISRR, vol. 70, no. 1, pp. 3–19, 2009.
[2]	Y. F. Chen, M. Liu, M. Everett, and J. P. How, “Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning.,” ICRA, pp. 285–292, 2017.
[3]	P. Long, W. Liu, and J. Pan, “Deep-Learned Collision Avoidance Policy for Distributed Multiagent Navigation.,” IEEE Robotics and Automation Letters, vol. 2, no. 2, pp. 656–663, 2017.
[4] http://gamma.cs.unc.edu/RVO2/

