---
title: 【图像处理】FAST角点检测
date: 2017-04-06 21:20:30
tags: 
    - image processing
    - local feature
categories: 【图像处理】
---

# 0 学习目标
* 理解FAST算法的原理
* 学习OpenCV中FAST角点检测的源码
* 利用OpenCV进行FAST的角点检测

# 1 原理

角点检测是很多视觉任务的第一步，比如说视觉SLAM，需要实时提取图像的特征点，并进行匹配。大牛们已经设计出很多角点检测算法，比如说SIFT(DOG),Harris和SUSAN等，这些角点检测算法可以提取出质量很高的角点，但是它们计算非常耗时，无法在SLAM这样对实时性要求很高的视觉任务中应用。于是，FAST角点检测算法在2006年被Edward Rosten和Tom Drummond提出[1]。 

下面对论文[1]中的算法原理进行简单的介绍：

<!--more-->

**FAST: Features from Accelerated Segment Test**
{% img [降维] http://on99gq8w5.bkt.clouddn.com/%E9%99%8D%E7%BB%B4.jpg?imageMogr2/thumbnail/300x300 降维示意图 %}

1. 对图像中的每一个候选点p,判断其是否为角点.首先，将它的像素值设为$I_p$.
2. 给定个阈值t.
3. 考虑候选点p周围圆形排列的16个像素点。
4. 判断候选点是否为角点的标准如下：
    * 如果16个像素点中有n个像素值大于$I_p+t$,或者小于$I_p-t$,那么p为角点；
    * 否则,p不是角点.
5. 为了快速排除大部分非角点，用上述的算法对其余的候选点进行检测。可以只考虑1,5,9,13个像素点：
    * 如果p为角点，上述四个点中至少有三个点的像素值大于$I_p+t$,或者小于$I_p-t$;
    * 否则,p不是角点.
   接着，用1-4中的方法对剩下的候选点进行角点检测.
6. 这种方法具有很好的性能，但是有下面四个问题：
    * 当n<12时,这种方法的扩展性不好；
    * 
    * 
    * 相邻的多个特征点会被检测到

**Machine Learning a Corner Detector**

上述问题中的前三个问题可以通过机器学习来解决，处理的过程分成了两个阶段，第一个阶段训练模型，第二个阶段利用决策树：
阶段一：
1. 选择一个图像集合进行角点学习(最好采用目标应用域中的图像集)
2. 对图像集中的每张图像上的每个像素点进行FAST角点检测,并用$x\in\{1\cdots16\}$来表示候选点周围的16个像素点；
3. 每个x相对于p来说有下面三种状态：
$$
\begin{eqnarray}
S_{p\to x}=
\begin{cases}
d,& I_{p\to x}\le I_p-t &(darker)\\
d,& I_p-t \le I_{p\to x} \le I_p+t &(similar)\\
b,& I_p+t \le I_{p\to x} &(brighter)
\end{cases}
\end{eqnarray}
$$
4. 将图像集中的所有像素点记为P，

阶段二：

# 2 源码解读

# 3 OpenCV实现FAST角点检测

# 4 参考文献


