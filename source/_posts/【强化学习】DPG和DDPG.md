---
title: 【强化学习】DPG和DDPG
date: 2017-11-08 14:48:58
tags:
    - RL
    - robotics
    - David Silver
categories: 【强化学习】
---

{% img [google-david-silver.png.jpeg] http://on99gq8w5.bkt.clouddn.com/google-david-silver.png.jpeg?imageMogr2/thumbnail/500x500 %}
<!--more-->
# 0 引言
为了解决连续动作空间的问题，也是绞尽了脑汁。D. Silver在2014和2016年分别提出了DPG[1]和DDPG[2]。就是封面的大神。

首先要区分两个概念：确定性策略和随机性策略。
* 随机性策略：$\pi_{\theta}(a|s)=\mathbb{P}[a|s;\theta]$
其含义是，在状态$s$时，动作符合参数为$\theta$的概率分布。比如说高斯策略:
$$
\pi_{\theta}(a|s)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(a-f_{\theta}(s))}{2\sigma^2})
$$
在状态$s$时，使用该策略获取动作，多次采样可以看到动作服从均值为$f_{\theta}(s)$，方差为$\sigma^2$的正太分布。也就是说，当使用随机策略时，虽然每次处于相同的状态，但是采取的动作也不一样。
* 确定性策略：$a=\mu_{\theta}(s)$
其含义是，对于相同的状态，确定性地执行同一个动作。

确定策略有哪些优点呢？**需要的采样数据少，算法效率高**

# 1 DPG
## 为什么要提出DPG？
在解决连续动作空间的问题上，确定性策略的求解通过动作值函数的梯度来实现，形式比较简单。相对随机性策略，估计起来也更加方便。
## 是什么？
确定性策略：$\mu_{\theta}:\mathcal{S}\to \mathcal{A}$,其中，$\theta$是策略函数的参数。目标就是确定$\theta$。

目标函数：
$$
J(\mu_{\theta})=\int_{\mathcal{S}}\rho^{\mu}(s)r(s,\mu_{\theta}(s))ds=\mathbb{E}_{s\sim \rho^{\mu}}[r(s,\mu_{\theta}(s))]
$$

其中，**状态转移的概率分布**可以表示为$p(s\to s^{\prime},t,\pi)$，即根据策略$\pi$从状态$s$经过$t$时间转移到状态$s^{\prime}$的概率分布。有了状态转移的概率分布，就可以定义**折扣的状态分布**
$$
\rho^{\pi}(s^{\prime}):=\int_{\mathcal{S}}\sum_{t=1}^{\infty}\gamma^{t-1}p_1(s)p(s\to s^{\prime},t,\pi)ds
$$
$p_1(s)$为初始的状态分布。

确定性策略梯度：
$$
\nabla_{\theta}J(\mu_{\theta})=\int_{\mathcal{S}}\rho^{\mu}(s)\nabla_{\theta}\mu_{\theta}(s)\nabla_{a}Q^{\mu}(s,a)\mid_{a=\mu_{\theta}(s)}ds  
$$
$$
=\mathbb{E}_{s\sim \rho^{\mu}}[\nabla_{\theta}\mu_{\theta}(s)\nabla_{a}Q^{\mu}(s,a)\mid_{a=\mu_{\theta}(s)}]
$$
为什么要对Q函数求偏导数呢？因为对于连续动作空间问题，使用贪婪法，每一步都需要求解概率最大的动作，相当于寻找Q函数的全局最大值，比较困难。于是，一个自然的想法就是将策略函数每次都沿着Q函数的梯度更新参数，而不是求解Q函数的全局最大值。于是，
$$
\theta^{k+1}=\theta^k+\alpha\mathbb{E}_{s\sim\rho^{\mu^k}}[\nabla_{\theta}Q^{\mu^k}(s,\mu_{\theta}(s))]
$$
根据链式法则，可以将上式改写成，
$$
\theta^{k+1}=\theta^k+\alpha\mathbb{E}_{s\sim\rho^{\mu^k}}[\nabla_{\theta}\mu_{\theta}(s)\nabla_{a}Q^{\mu^k}(s,a)\mid_{a=\mu_{\theta}(s)}]
$$


**同策略实现**
使用同一个策略产生训练数据，并提升这个策略，会导致“探索”不够充分，最终得到一个此优解。但是，为了展示算法的完整性，还是介绍一下。同样实现的过程用到了AC的思想，actor使用确定性策略梯度更新参数，critic使用Sarsa更新动作-状态值函数。同样的，状态值函数使用函数逼近的方式获得：$Q^w(s,a)$
TD-error: $\delta_t=r_t+\gamma Q^w(s_{t+1},a_{t+1})-Q^w(s_t,a_t)$
critic参数更新: $w_{t+1}=w_t+\alpha_w\delta_t\nabla_wQ^w(s_t,a_t)$
actor参数更新: $\theta_{t+1}=\theta_t+\alpha_{\theta}\nabla_{\theta}\mu_{\theta}(s_t)\nabla_{a}Q^w(s_t,a_t)\mid_{a=\mu_{\theta}(s)}$  (SGD)
**异策略实现**
实现大致和同策略一致，只不过学习的数据是通过一个随机策略生成的。

# 2 DDPG
## 为什么要提出DDPG？

DDPG解决了四大问题：
一是使用神经网络解决优化问题的一个前提是训练数据独立同分布，但是根据随机策略产生的数据不具备这样的属性，而且为了发挥硬件的特性，需要使用minibatch的数据来进行训练。借鉴DQN的思路，使用Experience Replay的思想。

使用双端队列的代码实现$^{[4]}$
```Python
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size      # 记忆容量
        self.count = 0                      # 统计当前记忆的条数
        self.buffer = deque()               # 双端队列实现记忆

    def add(self, s, a, r, t, s2):          # 存储一条记忆
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:   # 如果没有存满，队列后面增加一条
            self.buffer.append(experience)
            self.count += 1
        else:                               # 如果存满了，则弹出队列开头的一个，尾部增加一个
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):                         # 获取记忆的条数
        return self.count

    def sample_batch(self, batch_size):     # 从记忆中随机采样batch_size条记忆，以供学习
        '''     
        batch_size specifies the number of experiences to add 
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least 
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):                        # 清除记忆
        self.deque.clear()
        self.count = 0
```
二是直接使用神经网络实现Q-learning被证明是不稳定的，可能不会收敛。解决的办法是复制一份actor和critic网络作为target网络，使用相同的网络结构但是不同的参数。

三是当状态特征向量的纬数比较低，不同特征由于不同的单位，数值大小各不相同，并且随着环境的变化而发生变化，这时候就很难有效学习一组参数可以适用于各种环境。通常的解决办法是batch normalization，将每个minibatch中的样本的特征归一化到具有单位均值和方差。

四是增加探索的成分，获取全局最优解。
$$
\mu^{\prime}(s_t)=\mu(s_t\mid \theta^{\mu}_t)+\mathcal{N}
$$

## 是什么？
算法的伪代码如下：

{% img [DDPG] http://on99gq8w5.bkt.clouddn.com/DDPG.png?imageMogr2/thumbnail/600x600 %}

critic: $Q(s,a\mid\theta^{Q})$
$$
L=\frac{1}{N}\sum_i(y_i-Q(s_i,a_i\mid\theta^Q))^2
$$
其中，$y_i=r_i+\gamma Q^{\prime}(s_{i+1},\mu^{\prime}(s_{i+1}\mid\theta^{\mu^{\prime}})\mid \theta^{Q^{\prime}})$
actor: $\mu(s\mid\theta^{\mu})$
$$
\nabla_{\theta^{\mu}}J\approx\frac{1}{N}\sum_i\nabla_aQ(s,a\mid\theta^Q)\mid_{s=s_i,a=\mu(s_i)}\nabla_{\theta^{\mu}}\mu(s\mid\theta^{\mu})\mid_{s_i}
$$
actor: $\mu^{\prime}(s\mid\theta^{\mu^{\prime}})$
critic: $Q^{\prime}(s,a\mid\theta^{Q^{\prime}})$

两种更新策略：
1. 每隔C步更新一次；
2. 每次更新一点点，如伪代码所示，通常$\tau$是一个很小的数。

## DDPG代码实现$^{[4]}$
```Python
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp

from replay_buffer import ReplayBuffer

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(j))))
                break

def main(args):

    with tf.Session() as sess:

        env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess, env, args, actor, critic, actor_noise)

        if args['use_gym_monitor']:
            env.monitor.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
```


# 参考文献
[1] Deterministic Policy Gradients. D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra, M. Riedmiller. ICML 2014.
[2] Continuous Control with Deep Reinforcement Learning. T. Lillicrap, J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, D. Wierstra. ICLR 2016.
[3] 天津包子馅儿 强化学习知识大讲堂 知乎专栏
[4] http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
