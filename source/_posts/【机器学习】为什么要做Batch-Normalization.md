---
title: 【机器学习】为什么要做Batch Normalization?
date: 2017-11-03 15:16:20
tags:
    - NN
categories: 【机器学习】
---


之前介绍过**梯度爆炸**，就是说在误差反向传播的过程中，由于激活函数的导数值小于1，链式法则导致传过来的偏导越来越小，结果前面几层网络的权值得不到更新。

现在介绍的批量归一化是干什么的呢？
当f(x*w+b)，由于激活函数一般在正负一之间，当w和b固定的时候，x较小的时候，y=w*x+b也比较小，当x较大的时候，y就会处在饱和区。对于大值区分不出来。这就是NN遇到的问题，解决的办法就是批量归一化。
<!--more-->


tensorflow学习
基本概念
会话Session
```Python
# Method one
import tensorflow as tf
sess = tf.Session()
sess.run()
sess.close()

# Method two
with tf.Session() as sess:
    sess.run()
```
变量Variable
tf.Variable()
tf.constant()
tf.initialize_all_variables()

placeholder
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1, input2)

with tf.Session() as sess:
    sess.run(output, feed_dict={input1:[1.], input2:[3.]})
    