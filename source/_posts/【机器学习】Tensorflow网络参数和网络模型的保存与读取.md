---
title: 【机器学习】Tensorflow网络参数和网络模型的保存与读取
date: 2017-11-26 12:59:50
tags:
    - Tensorflow
categories: 【机器学习】
---
# 0 引言
在利用Caffe进行训练的时候，最终的训练结果会保存下来，在做预测的时候可以直接加载训练好的模型。但是目前接触的Tensorflow案例中，都是直接训练、然后在测试集上验证，最后退出整个程序。下次再使用的时候，就需要重新训练、预测。这样就很不科学。心想，肯定会有办法来保存和加载模型。于是，在莫烦的教程中看到保存和读取模型参数的教程[1]，他说Tensorflow初期，只支持网络参数的保存和读取，不能保存网络结构，如果想使用训练好的参数，必须重新搭建一模一样的网络结构，才能完成预测。但是后来可能Tensorflow觉得这样不方便，于是推出MetaGraph[2]，可以保存网络结构。本文主要介绍网络参数的保存和读取，以及网络结构的保存和读取。
<!--more-->
最主要的类：tf.train.Saver
初始化参数：
```Python
__init__(
    var_list=None,      # 需要保存变量的列表
    reshape=False,
    sharded=False,
    max_to_keep=5,  
    keep_checkpoint_every_n_hours=10000.0,
    name=None,
    restore_sequentially=False,
    saver_def=None,
    builder=None,
    defer_build=False,
    allow_empty=False,
    write_version=tf.train.SaverDef.V2,
    pad_step_number=False,
    save_relative_paths=False,
    filename=None
)
```
max_to_keep 参数用来设置保存模型的个数，默认为5，即保存最近的5个模型。如果想每训练一代epoch就想保存一次模型，可以将 max_to_keep设置为None或者0，如：saver=tf.train.Saver(max_to_keep=0)
方法：

# 1 网络参数的保存与读取
网络参数也就是Tensorflow中的Variables类型。

## 网络参数的保存
```Python
import tensorflow as tf

W = tf.Variable([[1.,2.,3.],[4.,5.,6.]], dtype=tf.float32, name='weights')
b = tf.Variable([[0.1,0.2,0.3]], dtype=tf.float32, name='bias')

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    save_path = saver.save(sess, 'model/params.ckpt')
    print("Saved to path: ", save_path)
```
运行上面的程序，会在当前文件夹下面创建model文件夹，并在model文件夹下，生成四个文件：checkpoint,params.ckpt.data-00000-of-00001,params.ckpt.index,params.ckpt.meta，它们的含义为：
* checkpoint文件保存了一个目录下所有的模型文件列表，这个文件是tf.train.Saver类自动生成且自动维护的。在 checkpoint文件中维护了由一个tf.train.Saver类持久化的所有TensorFlow模型文件的文件名。当某个保存的TensorFlow模型文件被删除时，这个模型所对应的文件名也会从checkpoint文件中删除。checkpoint中内容的格式为CheckpointState Protocol Buffer.
* params.ckpt.meta文件保存了TensorFlow计算图的结构，可以理解为神经网络的网络结构。TensorFlow通过元图（MetaGraph）来记录计算图中节点的信息以及运行计算图中节点所需要的元数据。TensorFlow中元图是由MetaGraphDef Protocol Buffer定义的。MetaGraphDef 中的内容构成了TensorFlow持久化时的第一个文件。保存MetaGraphDef 信息的文件默认以.meta为后缀名，文件model.ckpt.meta中存储的就是元图数据。
* params.ckpt.data-00000-of-00001文件保存了网络参数的值，但是数据是没有结构的。为了在网络中恢复模型，需要这样使用：
```Python
saver = tf.train.import_meta_graph(path_to_ckpt_meta)
saver.restore(sess, path_to_ckpt_data)
```
* params.ckpt.index还不清楚是干什么用的，猜想是一种映射关系。

## 网络参数的读取
```Python
import tensorflow as tf
import numpy as np

W = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name='bias')

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'model/params.ckpt')
    print("weights: ", sess.run(W))
    print("bias: ", sess.run(b))
```
# 2 网络模型的保存与读取
网络模型的导入导出是通过元图(MetaGraph)实现的。MetaGraph包含以下内容：
* MetaInfoDef for meta information, such as version and other user information.
* GraphDef for describing the graph.
* SaverDef for the saver.
* CollectionDef map that further describes additional components of the model, such as Variables, tf.train.QueueRunner, etc. In order for a Python object to be serialized to and from MetaGraphDef, the Python class must implement to_proto() and from_proto() methods, and register them with the system using register_proto_function.

## 导出
```Python
# Build the model
...
with tf.Session() as sess:
  # Use the model
  ...
# Export the model to /tmp/my-model.meta.
meta_graph_def = tf.train.export_meta_graph(filename='/tmp/my-model.meta')
```
## 导入
### 最简单的情况：
```Python
saver = tf.train.import_meta_graph(path_to_ckpt_meta)
saver.restore(sess, path_to_ckpt_data)
```
### 训练到一半，停下来，导入接着训练
```Python
...
# Create a saver.
saver = tf.train.Saver(...variables...)
# Remember the training_op we want to run by adding it to a collection.
tf.add_to_collection('train_op', train_op)
sess = tf.Session()
for step in xrange(1000000):
    sess.run(train_op)
    if step % 1000 == 0:
        # Saves checkpoint, which by default also exports a meta_graph
        # named 'my-model-global_step.meta'.
        saver.save(sess, 'my-model', global_step=step)
```
上面的训练，训练没有完成，停下来了，要接着训练：
```Python
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
  new_saver.restore(sess, 'my-save-dir/my-model-10000')
  # tf.get_collection() returns a list. In this example we only want the
  # first one.
  train_op = tf.get_collection('train_op')[0]
  for step in xrange(1000000):
    sess.run(train_op)
```
### 利用之前的训练结果，训练扩展后的网络模型
首先，定义一个网络，训练，保存结果
```Python
# Creates an inference graph.
# Hidden 1
images = tf.constant(1.2, tf.float32, shape=[100, 28])
with tf.name_scope("hidden1"):
  weights = tf.Variable(
      tf.truncated_normal([28, 128],
                          stddev=1.0 / math.sqrt(float(28))),
      name="weights")
  biases = tf.Variable(tf.zeros([128]),
                       name="biases")
  hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
# Hidden 2
with tf.name_scope("hidden2"):
  weights = tf.Variable(
      tf.truncated_normal([128, 32],
                          stddev=1.0 / math.sqrt(float(128))),
      name="weights")
  biases = tf.Variable(tf.zeros([32]),
                       name="biases")
  hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
# Linear
with tf.name_scope("softmax_linear"):
  weights = tf.Variable(
      tf.truncated_normal([32, 10],
                          stddev=1.0 / math.sqrt(float(32))),
      name="weights")
  biases = tf.Variable(tf.zeros([10]),
                       name="biases")
  logits = tf.matmul(hidden2, weights) + biases
  tf.add_to_collection("logits", logits)  # 装在参数集合里，方便后面使用

init_all_op = tf.global_variables_initializer()

with tf.Session() as sess:
  # Initializes all the variables.
  sess.run(init_all_op)
  # Runs to logit.
  sess.run(logits)
  # Creates a saver.
  saver0 = tf.train.Saver()
  saver0.save(sess, 'my-save-dir/my-model-10000')
  # Generates MetaGraphDef.
  saver0.export_meta_graph('my-save-dir/my-model-10000.meta')
```
然后，加载训练结果，扩展网络，接着训练。
```Python
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
  new_saver.restore(sess, 'my-save-dir/my-model-10000')
  # Addes loss and train.
  labels = tf.constant(0, tf.int32, shape=[100], name="labels")
  batch_size = tf.size(labels)
  labels = tf.expand_dims(labels, 1)
  indices = tf.expand_dims(tf.range(0, batch_size), 1)
  concated = tf.concat([indices, labels], 1)
  onehot_labels = tf.sparse_to_dense(
      concated, tf.stack([batch_size, 10]), 1.0, 0.0)
  logits = tf.get_collection("logits")[0]  # 提取网络参数
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=onehot_labels, logits=logits, name="xentropy")
  loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")

  tf.summary.scalar('loss', loss)
  # Creates the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(0.01)

  # Runs train_op.
  train_op = optimizer.minimize(loss)
  sess.run(train_op)
```

* .pd文件：the .pb file can save your whole graph (meta + data). To load and use (but not train) a graph in c++ you'll usually use it, created with freeze_graph, which creates the .pb file from the meta and data. Be careful, (at least in previous TF versions and for some people) the py function provided by freeze_graph did not work properly, so you'd have to use the script version. Tensorflow also provides a tf.train.Saver.to_proto() method, but I don't know what it does exactly.

# 参考文献
[1] Morvan tensorflow
[2] tensorflow.org python api

Tensorflow C++ 编译和调用图模型：http://blog.csdn.net/rockingdingo/article/details/75452711
