---
title: 【硬件设备】Jetson TX2深度学习库安装
date: 2017-11-01 15:59:45
tags:
    - Jetson TX2
    - deep learning
    - robotics
categories: 【硬件设备】
---

{% img [jetsontx2] http://on99gq8w5.bkt.clouddn.com/jetsontx2.png?imageMogr2/thumbnail/600x600 %}
<!--more-->
# 0 引言
购买

# 1 TensorFlow
google的库，都说NB。但是还没有学会怎么用，虽然已经买了TensorFlow实战。
```Bash
sudo pip install tensorflow-1.3.0-cp27-cp27mu-linux_aarch64.whl
```
tensorflow下载地址：
https://github.com/peterlee0127/tensorflow-tx2
# 2 caffe
最早听说的深度学习库，本科毕设一个同学做了一个图像分类任务。训练的模型用C++部署的时候用起来很方便。其他库的C++部署，还不会。
```Bash

```

# 3 mxnet
沐神带我飞
http://mxnet.incubator.apache.org/get_started/install.html
# 4 PyTorch
深度强化学习Atari的CNN+DQN使用过。
```Bash
#!/bin/bash
#
# pyTorch install script for NVIDIA Jetson TX1/TX2,
# from a fresh flashing of JetPack 2.3.1 / JetPack 3.0 / JetPack 3.1
#
# for the full source, see jetson-reinforcement repo:
#   https://github.com/dusty-nv/jetson-reinforcement/blob/master/CMakePreBuild.sh
#
# note:  pyTorch documentation calls for use of Anaconda,
#        however Anaconda isn't available for aarch64.
#        Instead, we install directly from source using setup.py
sudo apt-get install python-pip

# upgrade pip
pip install -U pip
pip --version
# pip 9.0.1 from /home/ubuntu/.local/lib/python2.7/site-packages (python 2.7)

# clone pyTorch repo
git clone http://github.com/pytorch/pytorch
cd pytorch
git submodule update --init

# install prereqs
sudo pip install -U setuptools
sudo pip install -r requirements.txt

# Develop Mode:
python setup.py build_deps
sudo python setup.py develop

# Install Mode:  (substitute for Develop Mode commands)
#sudo python setup.py install

# Verify CUDA (from python interactive terminal)
# import torch
# print(torch.cuda.is_available())
# a = torch.cuda.FloatTensor(2)
# print(a)
# b = torch.randn(2).cuda()
# print(b)
# c = a + b
# print(c)
```

# 5 theano
目前，对这个库没有什么感觉。只是同学在用来做DBN，听说过。
```Bash
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libblas-dev git
sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git --user  # Need Theano 0.8 or more recent

# test
# from theano import function, tensor
```

# 6 总结
