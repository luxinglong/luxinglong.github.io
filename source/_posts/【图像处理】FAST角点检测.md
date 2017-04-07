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

1. 对图像中的每一个候选点p,判断其是否为角点.首先，将它的像素值设为$I_p$.
2. 给定个阈值t.
3. 考虑候选点p周围圆形排列的16个像素点,如下图所示。
{% img [Snip20170407_1] http://on99gq8w5.bkt.clouddn.com/Snip20170407_1.png?imageMogr2/thumbnail/500x500 %}
4. 判断候选点是否为角点的标准如下：
    * 如果16个像素点中有n个像素值大于$I_p+t$,或者小于$I_p-t$,那么p为角点；
    * 否则,p不是角点.
5. 为了快速排除大部分非角点，用上述的算法对其余的候选点进行检测。可以只考虑1,5,9,13个像素点：
    * 如果p为角点，上述四个点中至少有三个点的像素值大于$I_p+t$,或者小于$I_p-t$;
    * 否则,p不是角点.
   接着，用1-4中的方法对剩下的候选点进行角点检测.
6. 这种方法具有很好的性能，但是有下面四个问题：
   问题一：当n<12时,这种方法的扩展性不好；
   问题二：快速检测算法对周边像素点的选择和排列隐含了角点外形的分布;
   问题三：对1，5，9，13四个像素点的测试得到的信息，在后面的角点检测中没有用到;
   问题四：相邻的多个角点会被检测到.

-------------------------------

**Machine Learning a Corner Detector**

上述问题中的前三个问题可以通过机器学习来解决，处理的过程分成了两个阶段，第一阶段构造候选点的特征向量，第二个阶段利用决策树进行分类：
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
4. 将图像集中的所有像素点记为P，选择一个特征x，计算P中所有点的特征值，根据这个特征可以将P划分为三个子集，$P_d,P_s,P_b$.
5. 定义一个布尔型变量$K_p$,如果p为角点，那么$K_p=1$,如果p不为角点，那么$K_p=0$.
阶段二：
1. 对候选点集P和特征向量集合，利用ID3算法根据最大信息增益生成决策树。
2. 候选点集合P的经验熵定义为：
$$
H(P)=(c+\bar{c})log_2(c+\bar{c})-clog_2c-\bar{c}log_2\bar{c}
$$
其中，$c=|\{p|K_p=1\}|$表示角点的数量，$\bar{c}=|\{p|K_p=0\}|$表示非角点的数量.
3. 特征x的信息增益定义为：
$$
H(P)-H(P_d)-H(P_s)-H(P_b)
$$
4. 根据最大信息增益原则选择出最能区分出角点和非角点的特征，将候选点集合分割成三个子集，在每个子集上依次进行上述的过程，直至子集的熵为0.
5. 生成的决策树可以作为一个角点检测器，来检测新的角点。

-------------------------------------------

**Non-maximal Suppression**
为了解决问题四，需要进行非最大抑制操作，但是因为FAST角点检测算法没有设计角点响应函数。因此首先定义角点响应函数，直观上看，有三种定义方式：
1. 保持p为角点的最大n;
2. 保持p为角点的最大阈值t;
3. 连续弧线上像素与中心像素进行像素值相减，取绝对值，并累加。

定义1、2的区分能力不够，为了加速定义3中角点响应函数的计算，需要进行修改。修改后的角点响应函数如下：
$$
V=max(\sum_{x\in S_bright}|I_{p\to x}-I_p|-t,\sum_{x\in S_dark}|I_p-I_{p\to x}|-t)
$$
其中，$S_bright=\{x|I_{p\to x}\ge I_p+t\}$
     $S_dark=\{x|I_{p\to x}\le I_p-t\}

有了角点响应函数，计算每个角点的响应函数值，应用非最大抑制，如果一个角点有一个响应函数值更高的相邻的角点，那么这个角点要去掉。
# 2 源码解读
{% codeblock hello.cpp %}
#include <iostream>
using namespace std;
int main(void)
{
    cout << "hello hexo" << endl;
    return 0;
}
{% endcodeblock %}
# 3 OpenCV实现FAST角点检测

# 4 参考文献
[1] Rosten E, Drummond T. Machine learning for high-speed corner detection[J]. Computer vision–ECCV 2006, 2006: 430-443.
[2]

