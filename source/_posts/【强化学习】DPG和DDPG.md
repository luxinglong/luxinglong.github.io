---
title: 【强化学习】DPG和DDPG
date: 2017-11-08 14:48:58
tags:
    - RL
    - robotics
categories: 【强化学习】
---

{% img [google-david-silver.png.jpeg] http://on99gq8w5.bkt.clouddn.com/google-david-silver.png.jpeg?imageMogr2/thumbnail/500x500 %}
<!--more-->
# 0 引言
为了解决连续动作空间的问题，也是绞尽了脑汁。D. Silver在2014和2016年分别提出了DPG和DDPG。就是开头的大神。

首先要区分两个概念：确定性策略和随机性策略。
* 随机性策略：$\pi_{\theta}(a|s)=\mathbb{P}[a|s;\theta]$
其含义是，在状态$s$时，动作符合参数为$\theta$的概率分布。比如说高斯策略:
$$
\pi_{\theta}(a|s)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(a-f_{\theta}(s))}{2\sigma^2})
$$
在状态$s$时，使用该策略获取动作，多次采样可以看到动作服从均值为$f_{\theta}(s)$，方差为$\sigma^2$的正太分布。也就是说，当使用随机策略时，虽然每次处于相同的状态，但是采取的动作也不一样。
* 确定性策略：$a=\eta_{\theta}(s)$
其含义是，对于相同的状态，确定性地执行同一个动作。

确定策略有哪些优点呢？
**需要的采样数据少，算法效率高**


# 1 DPG
为什么要提出DPG？
是什么？
效果怎么样？
评价一下：好的地方坏的地方

# 2 DDPG
为什么要提出DPG？
是什么？
效果怎么样？
评价一下：好的地方坏的地方

算法的伪代码如下：
{% img [DDPG] http://on99gq8w5.bkt.clouddn.com/DDPG.png?imageMogr2/thumbnail/500x500 %}


# 参考文献
[1] Deterministic Policy Gradients. D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra, M. Riedmiller. ICML 2014.
[2] Continuous Control with Deep Reinforcement Learning. T. Lillicrap, J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, D. Wierstra. ICLR 2016.
[3] 天津包子馅儿 强化学习知识大讲堂 知乎专栏
