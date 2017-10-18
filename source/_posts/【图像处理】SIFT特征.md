---
title: 【图像处理】SIFT特征
date: 2017-04-24 09:56:08
tags: 
    - image processing
    - local feature
categories: 【图像处理】
---
# 0 引言
Harris角点被誉为是局部特征领域中的第一篇文章，而SIFT特征则是最重要的一篇文章，现在的科研工作者正在向最后一篇文章努力。作为最重要的一篇论文提出的SIFT特征，之所以重要，是因为它具有尺度和旋转不变性，即使在仿射变换、视角变化、含有噪声和光照变化的条件下，特征匹配也具有鲁棒性。那么接下来，就分析一下，为什么SIFT具有这些特性？最后给出一些利用OpenCV实现的例子。

<!--more-->

# 1 原理
图像特征的计算可以通过滤波器级联的方式来实现，一般来说，计算越花时间的操作放在越后面。这样可以排除很多不需要计算的点。提高整体的计算效率。下面展示了计算图像特征的步骤：
1. 在图像的不同尺度空间检测极值点，选出兴趣点
2. 在每个兴趣点，。。。，选出关键点
3. 根据关键点周围的梯度信息，确定一个或者几个方向信息。将图像特征统一到相同的方向、尺度，这样特征就可以对方向和尺度具有不变性
4. 为每个关键点建立描述子，就是用一个特征向量来表示关键点，便于后面的相似度计算。

下面对上述的四个步骤，进行详细分析：
## 1.1 尺度空间的极值检测
首先，定义**尺度空间**:输入图像$I(x,y)$与高斯滤波器$G(x,y,\sigma)$卷积。如下式：
$$
L(x,y,\sigma)=G(x,y,\sigma)\ast I(x,y)
$$
其中，*号是卷积操作。高斯滤波器的形式为：
$$
G(x,y,\sigma)=\frac{1}{\sqrt{2\pi \sigma ^2}}e^{(x^2+y^2)/2\sigma ^2}
$$
为了提取兴趣点，需要回顾几个概念：
**梯度算子**：$\nabla f=(\frac{\partial f}{\partial x_1},\cdots,\frac{\partial f}{\partial x_n})$
**拉普拉斯算子**：$\triangle f=\nabla^2 f=\sum_{i=1}^n\frac{\partial^2f}{\partial x_i^2}$
下面推导LOG算子和DOG算子，并说明为什么GOG算子是LOG算子的近似。
**LOG算子**：
Gaussian kernel of width $\sigma$:
$$
G(x,y,\sigma)=\frac{1}{\sqrt{2\pi \sigma ^2}}e^{-(x^2+y^2)/2\sigma ^2}
$$
利用高斯平滑一副图像：
$$
\begin{equation}
\triangle [G(x,y,\sigma)\ast f(x,y)]=[\triangle G(x,y,\sigma)\ast f(x,y)]\\
                                    =LOG\ast f(x,y)\\
\frac{\partial}{\partial x}G(x,y,\sigma)=\frac{\partial}{\partial x}e^{-(x^2+y^2)/2\sigma ^2}=-\frac{x}{\sigma^2}^{-(x^2+y^2)/2\sigma ^2}\\
\frac{\partial^2}{\partial x^2}G(x,y,\sigma)=\frac{\partial}{\partial x}[-\frac{x}{\sigma^2}^{-(x^2+y^2)/2\sigma ^2}]=\frac{x^2-\sigma^2}{\sigma^4}e^{-(x^2+y^2)/2\sigma ^2}\\
LOG\triangleq\triangle G(x,y,\sigma)=\frac{\partial^2}{\partial x^2}G(x,y,\sigma)+\frac{\partial^2}{\partial y^2}G(x,y,\sigma)\\
                                    =\frac{x^2+y^2-2\sigma^2}{\sigma^4}e^{-(x^2+y^2)/2\sigma ^2}
\end{equation}
$$
下面展示DOG算子的推导。
**DOG算子**
$$
\begin{equation}
G(x,y,\sigma_1)=\frac{1}{\sqrt{2\pi \sigma_1 ^2}}e^{-(x^2+y^2)/2\sigma_1 ^2}\\
G(x,y,\sigma_2)=\frac{1}{\sqrt{2\pi \sigma_2 ^2}}e^{-(x^2+y^2)/2\sigma_2 ^2}\\
g(x,y,\sigma_1)=G(x,y,\sigma_1)\ast f(x,y)\\
g(x,y,\sigma_2)=G(x,y,\sigma_2)\ast f(x,y)\\
g(x,y,\sigma_1)-g(x,y,\sigma_2)=(G(x,y,\sigma_1)-G(x,y,\sigma_2))\ast f(x,y)\\
                    =DOG\ast f(x,y)
\end{equation}
$$

$$
DOG\triangleq G(x,y,\sigma_1)-G(x,y,\sigma_2)
$$


## 1.2 关键点的精确定位

## 1.3 
# 2 

# 参考文献
[1] Lowe D G. Distinctive image features from scale-invariant keypoints[J]. International journal of computer vision, 2004, 60(2): 91-110.
