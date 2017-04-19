---
title: 【机器视觉】基础矩阵F和本征矩阵E
date: 2017-04-19 13:11:54
tags:
    - Mutiple View Geometry
    - computer vision
categories: 【机器视觉】
---

# 0 学习目标
* 理解基础矩阵F和本征矩阵E的定义、区别和联系
* 学习基础矩阵F和本征矩阵E的计算
* 解析OpenCV相关源代码
* 利用OpenCV求解实现

<!--more-->

# 1 基本概念
## 对极几何(Epipolar geometry)
理解基础矩阵之前，首先要理解**对极几何**，因为基础矩阵就是对极几何的代数表示。对极几何在特征点匹配中发挥了很大的作用，可以极大减小匹配的搜索范围。对极几何可以用下图来描述：

{% img [epipolar_geometry]  http://on99gq8w5.bkt.clouddn.com/epipolar_geometry.png %}

几个对极几何中的概念：
**极点(epipole)：**相机光心连线与图像平面的交点$e$和$e^{\prime}$；
**极平面(epipolar plane)：**包含baseline的平面$\pi$；
**极线(epipolar line)：**极平面和图像平面相交的直线$l$和$l^{\prime}$。

对极几何关注的是空间三维点在两个图像平面上映射点之间的关系？也就是$x$和$x^{\prime}$之间的关系。

## 单应性矩阵和共面点成像
在计算机视觉中，平面的**单应性**被定义为一个平面到另外一个平面的投影映射。因此一个二维平面上的点映射到摄像机成像仪上的映射就是平面单应性的例子（如棋盘标定）。我们知道空间三维点$X=(X_w,Y_w,Z_w)$经过内参数矩阵K和外参数矩阵$[R|t]$可以投影到图像坐标系$x=(u,v)$。具体可以用下面的公式来表示：
{% img [camera] http://on99gq8w5.bkt.clouddn.com/camera.jpg %}
 > 此公式有个错误，要在第一个等号后面加$Z_C$.

而对于世界坐标系中共面的点，可以令其某一坐标为0，不妨设为$X=(X_w,Y_w,0)$，则单应性矩阵H可以用下面的公式推导：
{% img [homography] http://on99gq8w5.bkt.clouddn.com/homography.jpg %}
## 基础矩阵(Fundamental Matrix)
刚刚也说了基础矩阵就是对极几何的代数表示，也就是如何用代数的形式表示$x$和$x^{\prime}$之间的关系。图像平面上点$x$在另外一个平面上的对应点$x^{\prime}$在极线$l^{\prime}$上。而且存在着这样的映射关系$x \mapsto l^{\prime}$.这个映射关系可以通过两步来实现：
{% img [hpi] http://on99gq8w5.bkt.clouddn.com/hpi.png %}
步骤一：通过单应性矩阵H，将$x$映射到右图像平面$x^{\prime}$，即$x^{\prime}=Hx$.
步骤二：穿过$x^{\prime}$和$e^{\prime}$的直线就是$x$对应的极线 $l^{\prime}$。$l^{\prime}=e^{\prime}\times x^{\prime}=[e^{\prime}]_{\times}x^{\prime}=[e^{\prime}]_{\times}Hx=Fx$.

于是有，$l^{\prime}=Fx$.又因为$x^{\prime}$在直线$l^{\prime}$上，所以$x^{\prime T}l^{\prime}=x^{\prime T}Fx=0$.如果有很多点对满足这个公式，那么这些点对对应的三维空间点是共面的。

基础矩阵的性质有：秩为2，因为F将二维点映射到一条穿过$e^{\prime}$的直线上，即二维到一维。另外，$[e^{\prime}]_{\times}$秩为2， H秩为3. F是一个$3\times 3$的矩阵。

借用知乎大神一句话来通俗理解一下基础矩阵：
> 基础矩阵：反应了**空间一点X的像素点**在**不同视角摄像机**下**图像坐标系**中的表示之间的关系。即$(u_1,v_1,1)$之间$(u_2,v_2,1)$的关系。因此包含了内参数信息。

## 本征矩阵(Essential Matrix)



再次借用知乎大神一句话来通俗理解一下本征矩阵：
> 本征矩阵：反应了**空间一点X的像点**在**不同视角摄像机**下**摄像机坐标系**中的表示之间的关系。即$(X_C1,Y_C1,Z_C1)$之间$(X_C2,Y_C2,Z_C2)$的关系。因此不包含内参数信息。

# 2 基础矩阵F的计算
$x^{\prime T}Fx=0$这个关系的重要意义在于，我们不需要知道相机的内参数矩阵K，只通过两张图像的匹配点对变可以完成F的求解。那么至少需要多少个点对才能计算出F呢？


# 3 本征矩阵E的计算

# 4 OpenCV源码解析

# 5 实现

# 6 参考文献
[1] Hartley R, Zisserman A. Multiple View Geometry in Computer Vision Second Edition[J]. Cambridge University Press, 2000.
[2] OpenCV documentation, http://docs.opencv.org/2.4/index.html