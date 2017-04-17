---
title: 【图像处理】ORB特征
date: 2017-04-17 20:34:58
tags: 
    - image processing
    - local feature
categories: 【图像处理】
---

# 0 学习目标
* 理解ORB特征的原理和优缺点
* 利用OpenCV实现ORB特征
* 了解ORB特征在ORB-SLAM中的应用

# 1 原理
ORB (Oriented FAST and Rotated BRIEF)特征基于BRIEF特征，但是具有旋转不变性，并具有抗干扰的能力。ORB特征的贡献有：
* 为FAST角点增加了可以快速计算和精确的方向信息
* 

<!--more-->

## oFAST:FAST Keyoint Orientation
ORB特征点的提取是在FAST的基础上改进的，称为oFAST，也就是为每个FAST特征点增加一个方向信息，以此来使其具有旋转不变性。

**如何使其具有旋转不变性：**
回顾一下BRIEF描述子的计算过程：
在当前关键点P周围以一定模式选取N个点对，组合这N个点对的T操作的结果就为最终的描述子。当我们选取点对的时候，是以当前关键点为原点，以水平方向为X轴，以垂直方向为Y轴建立坐标系。当图片发生旋转时，坐标系不变，同样的取点模式取出来的点却不一样，计算得到的描述子也不一样，这是不符合我们要求的。因此我们需要重新建立坐标系，使新的坐标系可以跟随图片的旋转而旋转。这样我们以相同的取点模式取出来的点将具有一致性。打个比方，我有一个印章，上面刻着一些直线。用这个印章在一张图片上盖一个章子。印章不变动的情况下，转动下图片，再盖一个章子，但这次取出来的点对就和之前的不一样。为了使2次取出来的点一样，我需要将章子也旋转同一个角度再盖章。ORB在计算BRIEF描述子时建立的坐标系是以关键点为圆心，以关键点和取点区域的形心的连线为X轴建立2维坐标系。
{% img [orientation] http://on99gq8w5.bkt.clouddn.com/orientation.png %}
P为关键点。圆内为取点区域，每个小格子代表一个像素。现在我们把这块圆心区域看做一块木板，木板上每个点的质量等于其对应的像素值。根据积分学的知识我们可以求出这个密度不均匀木板的质心Q。计算公式如下。其中R为圆的半径。
$$
m_{pq}=\sum_{x=-R}^R\sum_{y=-R}^Rx^py^qI(x,y)
$$
$$
C=(m_{10}/m_{00},m_{01}/m_{00})
$$
$$
\theta=atan2(m_{01},m_{10});
$$
我们知道圆心是固定的而且随着物体的旋转而旋转。当我们以PQ作为坐标轴时，在不同的旋转角度下，我们以同一取点模式取出来的点是一致的。这就解决了旋转一致性的问题。

**如何解决对噪声敏感的问题：**
BRIEF使用的是pixel跟pixel的大小来构造描述子的每一个bit。这样的后果就是对噪声敏感。因此，在ORB的方案中，做了这样的改进，不再使用pixel-pair，而是使用9×9的patch-pair，也就是说，对比patch的像素值之和。（可以通过积分图快速计算）。

## rBRIEF:Rotation-Aware BRIEF



## 优点
1. 具有旋转不变性
2. 性能与SIFT接近，但是计算比SIFT快2的阶数次方

## 缺点
1. 不具备尺度不变性

# 2 源代码解析

# 3 OpenCV实现

# 4 参考文献
[1] Rublee E, Rabaud V, Konolige K, et al. ORB: An efficient alternative to SIFT or SURF[C]//Computer Vision (ICCV), 2011 IEEE International Conference on. IEEE, 2011: 2564-2571.
[2] OpenCV documentation, http://docs.opencv.org/2.4/index.html