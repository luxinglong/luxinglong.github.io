---
title: 【强化学习】演员-评论家算法 Actor-Critic
date: 2017-11-05 22:17:18
tags:
    - RL
    - robotics
categories: 【强化学习】
---

{% img [AC1] http://on99gq8w5.bkt.clouddn.com/AC1.png?imageMogr2/thumbnail/500x500 %}
<!--more-->
# 0 引言
同时用函数逼近策略和值函数，可以用“演员-评论家”算法来实现。对于连续动作空间，为什么不直接使用PG，非要折腾出一个Actor-Critic算法。

它的优点是：
* 相对于Q-learning来说，收敛性更好
* 相对于PG来说，可以单步更新参数，而不用回合更新，提高了效率。

# 1 AC
首先，介绍什么是演员评论家算法。

AC算法来自策略梯度。
策略梯度公式：
$$
\Delta_{\theta}J(\theta)=\mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}log\pi_{\theta}(s,a)Q^{\pi_{\theta}}(s,a)]
$$
第一部分$\nabla_{\theta}log\pi_{\theta}(s,a)$决定参数调整的方向，第二部分$Q^{\pi_{\theta}}(s,a)$决定调整的幅度。AC算法的思想就是把第二部分独立出来，使用另外一个网络来表示。也就是评论家部分。保留第一部分为演员部分。

然后，介绍两个部分如何更新自己的参数。引用一张网上经典的图：
{% img [actor-critic] http://on99gq8w5.bkt.clouddn.com/actor-critic.jpg?imageMogr2/thumbnail/400x400 %}

Actor部分，使用策略梯度更新参数，不同的是，策略梯度的第二部分换成一个带有参数的$Q^w(s,a)$。
Critic部分，使用策略评估的方式更新参数，如TD。$Q^w(s,a) \approx Q^{\pi_{\theta}}(s,a)$

图中的TD-error是个什么鬼？就是TD-target减去V的估计值。

问题：引入Critic部分，会不会引入偏差？理论证明只要满足下面两个条件就不会引入偏差。
1. $Q^w(s,a)=\nabla_{\theta}log\pi_{\theta}(s,a)^Tw$
2. 参数$w$由最小化均方误差$\epsilon^2(w)=\mathbb{E}_{s\sim\rho^{\pi},a\sim\pi_{\theta}}[(Q^w(s,a)-Q^{\pi_{\theta}}(s,a))^2]$获得。

那么Actor部分的梯度误差便成了：
$$
\Delta_{\theta}J(\theta)=\mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}log\pi_{\theta}(s,a)Q^w(s,a)]
$$

根据第二部分，可以将AC算法总结如下：
{% img [PGs] http://on99gq8w5.bkt.clouddn.com/PGs.png?imageMogr2/thumbnail/500x500 %}

# 2 QAC
伪代码为：

{% img [qac] http://on99gq8w5.bkt.clouddn.com/qac.png?imageMogr2/thumbnail/500x500 %}

**advantage function**
对于一个策略$\pi_{\theta}$来说，advantage function定义为
$$
A^{\pi_{\theta}}(s,a)=Q^{\pi_{\theta}}(s,a)-V^{\pi_{\theta}}(s)
$$
则策略梯度可以表示成：
$$
\nabla_{\theta}J(\theta)=\mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}log\pi_{\theta}(s,a)A^{\pi_{\theta}}(s,a)]
$$

使用advantage function的好处在于：
* 减小策略梯度的方差
* critic可以很好地估计出advantage function
* 用两套参数分别估计状态值函数和动作-状态值函数
$$
V_v(s) \approx V^{\pi_{\theta}}(s)
$$
$$
Q_w(s,a) \approx Q^{\pi_{\theta}}(s,a)
$$
则advantage function为
$$
A(s,a) = Q_w(s,a) - V_v(s)
$$

# 3 算法实现
设计两个类，一个Actor，另一个Critic。首先，介绍离散动作空间的问题，CartPole
Actor的输入是状态、输出是动作。
```Python
class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state") # 一个状态
        self.a = tf.placeholder(tf.int32, None, "act")                # 一个动作
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        # 三层网络结构：输入层+隐含层+输出层(softmax)
        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        # 损失函数
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int
```
接着定义Critic类如下：
Critic的输入是状态，输出是TD-error。
```Python
class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")  # 一个状态
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")         # 价值
        self.r = tf.placeholder(tf.float32, None, 'r')                 # 即时回报

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        # 定义损失函数，TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error
```
最后就是学习的过程
```Python
for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        # 关键的两步，单步更新
        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break
```
**CartPole-v0实现**

如果是连续动作空间，这时候Actor类就需要改写
```Python
class Actor(object):
    def __init__(self, sess, n_features, action_bound, lr=0.0001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.float32, None, name="act")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")  # TD_error

        l1 = tf.layers.dense(
            inputs=self.s,
            units=30,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l1'
        )

        mu = tf.layers.dense(
            inputs=l1,
            units=1,  # number of hidden units
            activation=tf.nn.tanh,  # 激活函数发生了变化，连续输出
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='mu'
        )

        sigma = tf.layers.dense(
            inputs=l1,
            units=1,  # output units
            activation=tf.nn.softplus,  # get action probabilities 不知道什么鬼
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(1.),  # biases
            name='sigma'
        )
        global_step = tf.Variable(0, trainable=False)
        # self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)
        self.mu, self.sigma = tf.squeeze(mu*2), tf.squeeze(sigma+0.1)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1])

        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)  # loss without advantage
            self.exp_v = log_prob * self.td_error  # advantage (TD_error) guided loss
            # Add cross entropy cost to encourage exploration
            self.exp_v += 0.01*self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)    # min(v) = max(-v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.s: s})  # get probabilities for all actions
```
其他部分不变。
**Pendulum-v0实现**

# 参考文献
[1] David Silver, reinforcement learning lecture 7
[2] Morvan 强化学习教程 AC
