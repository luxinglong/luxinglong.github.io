---
title: 【强化学习】算法实践-DQN
date: 2017-11-01 14:57:01
tags:
    - RL
    - robotics
    - python
categories: 【强化学习】
---

{% img [dqn] http://on99gq8w5.bkt.clouddn.com/dqn.png?imageMogr2/thumbnail/600x600 %}
<!--more-->
# 0 引言
利用神经网络对值函数或动作-状态值函数进行逼近的做法，从90年代开始就有了。但是那个时候总是会出现不稳定、不收敛的情况，因此并没有得到很多的应用。直到DeepMind在2015年出手，利用DQN解决了这个难题。

# 1 抽象Agent基类
Sarsa和Sarsa($\lambda$)所用的Agent类具有执行一个策略、执行一个动作和学习三个主要方法。但是，在连续状态空间的问题中，需要用到值函数逼近，那么就要求Agent具备记忆一定数量已经经历过的状态转换对象的功能，另外还要具备从记忆中随机获取一定数量的状态转换对象以供批量学习的功能。为此，Agent基类设计如下：

```Python
class Agent(object):
    '''Base Class of Agent
    '''

    def __init__(self, env = None, trans_capacity = 0):
        # 保存一些Agent可以观测到的环境信息以及已经学到的经验
        self.env = env
        self.obs_space = env.observation_space if env is not None else None
        self.action_space = env.action_space if env is not None else None

        # 表示Agent的记忆内容
        self.experience = Experience(capacity = trans_capacity)

        # 记录Agent当前的状态。注意要对该变量的维护和更新
        self.state = None
    
    def performPolicy(self, policy_fun, s):
        if policy_fun is None:
            return self.action_space.sample()
        return policy_fun(s)

    def act(self, a0):
        s0 = self.state
        s1, r1, is_done, info = self.env.step(a0)
        # TODO: add extra code here
        trans = Transition(s0, a0, r1, is_done, s1)
        total_reward = self.experience.push(trans)
        self.state = s1
        return s1, r1, is_done, info, total_reward

    def learning(self):
        '''need to be implemented by all subclasses
        '''

        raise NotImplementedError

    def sample(self, batch_size = 64):
        '''随机取样
        '''
        return self.experience.sample(batch_size)

    # @property在这里是方法修饰器，将total_trans方法转换成属性，
    # 可以通过agent.total_trans来访问
    @property 
    def total_trans(self):
        '''得到Experience里记录的总的状态转换数量
        '''
        return self.experience.total_trans
```

## agent相关概念的建模
**状态转换(Transition)类**
状态转换记录了：agent的当前状态s0、agent在当前状态下执行的动作a0、个体在状态s0时执行a0后环境反馈的即时奖励值reward以及新状态s1，此外用一个bool型变量记录状态s1是不是一个终止状态，以此表明包含该状态转换的Episode是不是一个完整的Episode。

```Python
class Transition(object):
    def __init__(self, s0, a0, reward:float, is_done:bool, s1):
        self.data = [s0, a0, reward, is_done, s1]
    
    def __iter__(self):
        return iter(self.data)

    def __str__(self):
        return "s:{0:<3} a:{1:<3} r:{2:<4} is_end:{3:<5} s1:{4:<3}".\
            format(self.data[0],
                   self.data[1],
                   self.data[2],
                   self.data[3],
                   self.data[4])
    @property
    def s0(self): return self.data[0]
    @property
    def a0(self): return self.data[1]
    @property
    def reward(self): return self.data[2]
    @property
    def is_done(self): return self.data[3]
    @property
    def s1(self): return self.data[4]
```
**场景片段(Episode)类**
Episode类的主要功能是记录一系列的Episode，这些Episode就是由一系列的有序Transition对象构成，同时为了便于分析，我们额外增加一些功能，比如在记录一个Transition对象的同时，累加其即时奖励值以获得个体在经历一个Episode时获得的总奖励；又比如我们可以从Episode中随机获取一定数量、无序的Transition，以提高离线学习的准确性；此外由于一个Episode是不是一个完整的Episode在强化学习里是一个非常重要的信息，为此特别设计了一个方法来执行这一功能。
```Python
class Episode(object):
    def __init__(self, e_id = 0):
        self.total_reward = 0  # 总的获得的奖励
        self.trans_list = []   # 状态转移列表
        self.name = str(e_id)  # 可以给Episode取一个名字：“闯关成功、黯然失败？”

    def push(self, trans):
        self.trans_list.append(trans)
        self.total_reward += trans.reward
        return self.total_reward

    @property
    def len(self):
        return len(self.trans_list)
    
    def __str__(self):

    def print_detail(self):
        print("detail of ({0}):".format(self))
        for i,trans in enumerate(self.trans_list):
            print("step{0:<4}".format, end=" ")
            print(trans)

    def pop(self):
        if self.len > 1:
            trans = self.trans_list.pop()
            self.total_reward -= trans.reward
            return trans
        else:
            return None

    def is_complete(self):
        ''' 检测一个episode是不是完整的episode
        '''
        if self.len == 0:
            return False
        return self.trans_list[self.len - 1].is_done

    def sample(self,batch_size = 1):
        ''' 随机产生一个trans
        '''
        return random.sample(self.trans_list, k = batch_size)

    def __len__(self):
        return self.len

```
**经历(Experience)类**
一个个Episode组成了agent的经历(Experience)。也有一些模型使用一个叫“Memory”的概念来记录agent个体既往的经历，其建模思想是Memory仅无序存储一系列的Transition，不使用Episode这一概念，不反映Transition对象之间的关联，这是可以完成基于记忆的离线学习的强化学习算法的，甚至其随机采样过程更简单。但是文献还是跟随[2]的思想，使用Episode作为中间数据结构，以后根据实践的情况再做调整。

一般来说，经历或者记忆的容量是有限的，为此需要设定一个能够表示记录Transition对象的最大上限，称为容量(capacity).一旦agent经历的Transition数量超过该容量，则将抹去最早期的Transition，为最近期的Transition腾出空间。可以想象，一个Experience类应该至少具备如下功能：移除早期的Transition；记住一个Transition；从Experience中随机采样一定数量的Transition。

```Python
class Experience(object):
    '''this class is used to record the whole experience of an agent organized
    by an episode list. agent can randomly sample transitions or episode from
    its experience.
    '''
    def __init__(self, capacity = 20000):
        self.capacity = capacity   # 容量：指的是trans总数量
        self.episodes = []         # episode列表
        self.next_id = 0           # 下一个episode的ID
        self.total_trans = 0       # 总的状态转换数量
    
    def __str__(self):   # python中的特殊方法，以字符串的形式返回对象的信息，调用方式，print experience 
        return "exp info:{0:5} episodes, memory usage {1}/{2}".\
                format(self.len, self.total_trans, self.capacity)

    def __len__(self):   # python中的特殊方法，返回集合中所含项目的数量，即Experience中episode的数量
        return self.len

    @property
    def len(self):
        return len(self.episodes)

    def _remove(self, index = 0): # 私有方法，类外方法不可以访问
        '''扔掉一个Episode，默认第一个
           remove an episode, default the first one
           args:
            the index of the episode to remove
           return:
            if exists return the episode else return None
        '''
        if index > self.len - 1:
            raise(Exception("invalid index"))
        if self.len > 0:
            episode = self.episodes[index]
            self.episodes.remove(episode)
            self.total_trans -= episode.len
            return episode
        else:
            return None
    
    def _remove_first(self):
        self._remove(index = 0)

    def push(self, trans):
        '''压入一个状态转换
        '''
        if self.capacity <= 0:
            return
        while self.total_trans >= self.capacity: 
            episode = self._remove_first()
        cur_episode = None
        if self.len == 0 or self.episodes[self.len-1].is_complete():
            cur_episode = Episode(self.next_id)
            self.next_id += 1
            self.episodes.append(cur_episode)
        else:
            cur_episode = self.episode[self.len-1]
        self.total_trans += 1
        return cur_episode.push(trans)   # return total reward of an episode

    def sample(self, batch_size = 1):
        '''randomly sample some transitions from agent's experience.abs
        随机获取一定数量的Transition对象
        args:
            number of transitions need to be sampled
        return:
            list of Transition
        '''
        sample_trans = []
        for _ in range(batch_size):
            index = int(random.random() * self.len)
            sample_trans += self.episodes[index].sample()
        return sample_trans

    def sample_episode(self, episode_num = 1):
        '''随机获取一定数量完整的Episode
        '''
        return random.sample(self.episodes, k = episode_num)

    @property
    def last(self):
        if self.len > 0:
            return self.episodes[self.len-1]
        return None

```
# 2 PuckWorld
在[1]中第七讲提到了PuckWorld，就是让一个“搞怪的小妖精”去抓取目标，目标随机出现在区域的一个位置，并且每隔30秒刷新一次。
{% img [puckworld] http://on99gq8w5.bkt.clouddn.com/puckworld.png?imageMogr2/thumbnail/400x400 %}
PuckWorld问题的强化学习模型描述如下：
状态空间：puck的位置、速度和目标的位置，即{$p_x,p_y,v_x,v_y,d_x,d_y$}
动作空间：上、下、左、右以及不作为
奖励函数：离目标越近，奖励值越大

下面展示PuckWorld的实现：
```Python
class PuckWorldEnv(gym.Env): 
    
    def __init__(self):
        self.width = 600          # 场景的宽度
        self.height = 600         # 场景的长度
        self.l_unit = 1.0         # 场景的长度单位
        self.v_unit = 1.0         # 速度单位
        self.max_speed = 0.025    # agent沿着某一个轴的最大速度

        self.re_pos_interval = 30 # 目标重置的时间间隔
        self.accel = 0.002        # agent的加速度
        self.rad = 0.05           # agent半径
        self.target_rad = 0.01    # 目标半径
        self.goal_dis = self.rad  # 目标接近距离
        self.t = 0                # PuckWorld时钟
        self.update_time = 100    # 目标重置时间

        self.low = np.array([0,                # agent px 的最小值 
                             0,                # agent py 的最小值 
                             -self.max_speed,  # agent vx 的最小值 
                             -self.max_speed,  # agent vy 的最小值 
                             0,                # 目标 dx 的最小值
                             0])               # 目标 dy 的最小值

        self.high = np.array([self.l_uint,
                              self.l_uint,
                              self.max_speed,
                              self.max_speed,
                              self.l_uint,
                              self.l_uint])   

        self.reward = 0
        self.action = None
        self.viewer = None

        # 0,1,2,3,4 分别表示左、右、上、下和静止不动五个动作
        self.action_space = spaces.Discrete(5)
        # 观察空间由low和high决定
        self.observation_space = spaces.Box(self.low, self.high)   

        self._seed()
        self.reset()  
```
“物理引擎”实现如下：
```Python
def _step(self, action):
    assert self.action_space.contains(action), \
        "%r (%s) invalid" % (action, type(action))

    self.action = action

    ppx,ppy,pvx,pvy,tx,ty = self.state  # 获取当前状态
    ppx,ppy = ppx+pvx, ppy+pvy    # 更新agent的位置
    pvx,pvy = pvx*0.95, pvy*0.95  # natural velocity loss

    if action == 0: pvx -= self.accel  # 左
    if action == 1: pvx += self.accel  # 右
    if action == 2: pvy += self.accel  # 上
    if action == 3: pvy -= self.accel  # 下
    if action == 4: pass               # 不动

    # 碰壁处理
    if ppx < self.rad:
        pvx *= -0.5
        ppx = self.rad
    if ppx > 1 - self.rad:
        pvx *= -0.5
        ppx = 1 - self.rad
    if ppy < self.rad:
        pvy *= -0.5
        ppy = self.rad
    if ppy > 1 - self.rad
        pvy *= -0.5
        ppy = 1 - self.rad

    self.t += 1
    if self.t %  self.update_time == 0:
        tx = self._random_pos()
        ty = self._random_pos()

    dx, dy = ppx - tx, ppy - ty
    dis = self._compute_dist(dx, dy)

    self.reward = self.goal_dis - dis  # agent半径减去中心距离

    done = bool(dis <= self.goal_dis)

    self.state = (ppx, ppy, pvx, pvy, tx, ty)
    return np.array(self.state), self.reward, done, {}

def _random_pos(self):
    return self.np_random.uniform(low = 0, high = self.l_uint)

def _compute_dis(self, dx, dy):
    return math.sqrt(math.pow(dx,2)+math.pow(dy,2))
```
“图像引擎”实现如下：
```Python
def _render(self, mode='human', close=False):
    if close:
        if self.Viewer is not None:
            self.viewer.close()
            self.viewer = None
        return 

    scale = self.width / self.l_uint  # 计算两者的映射关系
    rad = self.rad * scale            # 用像素尺寸来描述agent的半径
    t_rad = self.target_rad * scale   # 用像素尺寸来描述目标的半径

    if self.Viewer is None:
        from gym.env.classic_control import rendering
        self.viewer = rendering.Viewer(self.width, self.height)

        # 绘制目标和黑边框
        target = rendering.make_circle(t_rad, 30, True)
        target.set_color(0.1, 0.9, 0.1)
        self.viewer.add_geom(target)
        target_circle =  rendering.make_circle(t_rad, 30, False)
        target_circle.set_color(0, 0, 0)
        self.viewer.add_geom(target)
        self.target_trans = rendering.Transform()
        target.add_attr(self.target_trans)
        target_trans.add_attr(self.target_trans)

        # 绘制agent和黑边框
        self.agent = rendering.make_circle(rad, 30, True)
        self.agent.set_color(0, 1, 0)
        self.viewer.add_geom(self.agent)
        self.agent_trans = rendering.Transform()
        self.agent.add_attr(self.agent_trans)
        agent_circle = rendering.make_circle(rad, 30, True)
        agent_circle.set_color(0, 0, 0)
        agent_circle.add_attr(self.agent_trans)
        self.viewer.add_geom(agent_circle)

        self.line_trans = rendering.Transform()
        self.arrow = rendering.FilledPolygon([
            (0.7*rad, 0.15*rad),
            (rad,0),
            (0.7*rad, -0.15*rad) 
        ])
        self.arrow.set_color(0, 0, 0)
        self.arrow.add_attr(self.line_trans)
        self.viewer.add_geom(self.arrow)
    
    # 更新目标和小车的位置
    ppx,ppy,_,_,tx,ty = self.state
    self.target_trans.set_translation(tx*scale, ty*scale)
    self.agent_trans.set_translation(ppx*scale, ppy*scale)

    # 按距离给agent着色
    vv, ms = self.reward + 0.3, 1 
    r, g, b = 0, 1, 0
    # vv >= 0 表示agent距离target比较近，越大越近，最大可以达到0.35，则 0.65\1\0.65 很绿
    # vv < 0 表示agent距离target比较远，越小越远，最小可以达到-1，则 1\0\0 很红
    if vv >= 0:
        r, g, b = 1 - vv*ms, 1, 1 - vv*ms
    else:
        r, g, b = 1, 1 + ms*vv, 1 + ms*vv
    self.agent.set_color(r, g, b)

    a = self.action
    if a in [0, 1, 2, 3]:
        # 根据action绘制箭头
        if a == 0: degree = 180
        elif a == 1: degree = 0
        elif a == 2: degree = 90
        else: degree = 270

        self.line_trans.set_translation(ppx*scale, ppy*scale)
        self.line_trans.set_rotation(degree/RAD2DEG)
        self.arrow.set_color(0, 0, 0)
    else:
        self.arrow.set_color(r,g,b)
    
    return self.viewer.render(return_rgb_arrgy = mode == 'rgb_array')
```
"重置函数"实现：
```Python
def _reset(self):
    self.state = np.array([
        self._random_pos(),
        self._random_pos(),
        0,
        0,
        self._random_pos(),
        self._random_pos()
    ])
    return self.state
```

# 3 DQN算法实现

{% img [dqn_a] http://on99gq8w5.bkt.clouddn.com/dqn_a.png?imageMogr2/thumbnail/500x500 %}

把使用神经网络近似表示价值函数的功能封装到一个Approximator类中，然后再实现包含此价值函数的继承自Agent基类的个体类：ApproxQAgent，最后观察其在PuckWorld和CartPole环境中的训练效果。基于深度学习的部分使用PyTorch库。

**Approximator类的实现**
Approximator类作为价值函数的近似函数，其要实现的功能很简单：一是输出基于一个状态-动作对$\langle s,a \rangle$在参数w描述的环境下的价值$Q(s,a,w)$；二是调整参数来更新状态-动作对$\langle s,a \rangle$的价值。

在本例中，使用第三种值函数逼近方式，也就是输入为状态，输出为s与不同动作组成的状态-动作值函数值。在__init__构造函数中声明基于一个隐含层的神经网络。
```Python
# -*- coding:utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import copy

class Approximator(torch.nn.Module):
    def __init__(self, dim_input = 1, dim_ouput = 1, dim_hidden = 16):
        super(Approximator, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        
        self.linear1 = torch.nn.Linear(self.dim_input, self.dim_hidden)
        self.linear2 = torch.nn.Linear(self.dim_hidden, self.dim_output)
```

前向传输，预测状态x对应的价值：
```Python
def _forward(self, x):
    h_relu = self.linear1(x).clamp(min=0)
    y_pred = self.linear2(h_relu)
    return y_pred
```

利用fit方法来训练，更新网络参数：
```Python
def fit(self,x,
             y,
             criterion=None,
             optimizer=None,
             epochs=1,
             learning_rate=1e-4):

if criterion is None:
    criterion = torch.nn.MSELoss(size_average = False)
if optimizer is None:
    optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
if epochs < 1:
    epochs = 1

x = self._prepare_data(x)
y = self._prepare_data(y,False)

for t in range(epochs):
    y_pred = self._forward(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

return loss
```

还需要设计一个方法_prepare_data来对输入数据进行一定的修饰
```Python
def _prepare_data(self, x, requires_grad = True):
'''将numpy格式的数据转化为Torch的Variable
'''
    if isinstance(x, np.ndarray):
        x = Variable(torch.from_numpy(x), requires_grad = requires_grad)
    if isinstance(x, int):
        x = Variable(torch.Tensor([[x]]), requires_grad = requires_grad)

    x = x.float() 
    if x.data.dim() == 1:
        x = x.unsqueeze(0)
    return x
```

同时，为了使得agent在使用近似函数时更加简洁，可以为Approximator类写一个__call__方法，使得可以像执行函数一样来使用该类提供的方法：
```Python
def __call__(self, x):
    '''return an output given input. similar to predict function
    '''
    x = self._prepare_data(x)
    pred = self._foward(x)
    return pred.data.numpy()
```

最后一个比较重要的事情，由于一些高级的DQN算法使用两个近似函数+经验回放的机制来训练agent，因此会产生将一个近似函数的神经网络参数拷贝给另一个近似函数的神经网络的过程，也就是拷贝网络的过程，我们也需要提供一个能完成此功能的方法：
```Python
def clone(self):
    '''返回当前模型的深度拷贝对象
    '''
    return copy.deepcopy(self)
```

**ApproxQAgent类的实现**
构造函数：
```Python
class ApproxQAgent(Agent):
    '''使用近似的价值函数实现的Q学习个体
    '''
    def __init__(self, env=None,
                       trans_capacity = 20000,
                       hidden_dim = 16):
        if env is None:
            raise "agent should have an enviornment"
        super(ApproxQAgent, self).__init__(env, trans_capacity)
        self.input_dim, self.output_dim = 1, 1

        # 适应不同的状态和行为空间类型
        if isinstance(env.observation_space, spaces.Discrete):
            self.input_dim = 1
        elif isinstance(env.observation_space, spaces.Box):
            self.input_dim = env.observation_space.shape[0]

        if isinstance(env.action_space, spaces.Discrete):
            self.input_dim = env.action_space.n
        elif isinstance(env.action_space, spaces.Box):
            self.input_dim = env.action_space.shape[0]

        # 隐藏层神经元的数量
        self.hidden_dim = hidden_dim

        # 关键在下面两句，声明了两个近似价值函数
        # 变量Q是一个策略评估网络
        # 该网络参数在一定时间段不更新参数
        self.Q = Approximator(dim_input = self.input_dim,
                              dim_output = self.output_dim,
                              dim_hidden = self.hidden_dim)
        # 变量PQ是一个策略近似网络
        # 该网络参数每一步都会更新
        self.PQ = self.Q.clone()
```
从经历中学习：
```Python
def _learn_from_memory(self, gamma, batch_size, learning_rate, epochs):
    
    # 随机获取记忆里的batch_size个Transmition，从基类Agent继承来的方法
    trans_pieces = self.sample(batch_size)

    states_0 = np.vstack([x.s0 for x in trans_pieces])
    actions_0 = np.array([x.a0 for x in trans_pieces])
    reward_1 = np.array([x.reward for x in trans_pieces])
    is_done = np.array([x.is_done for x in trans_pieces])
    states_1 = np.vstack([x.s1 for x in trans_pieces])

    X_batch = states_0
    y_batch = self.Q(states_0)

    # 使用了Batch
    Q_target = reward_1 + gamma * np.max(self.Q(states_1),axis=1)*(~is_done)
    y_batch[np.arange(len(X_batch)),actions_0] = Q_target

    loss = self.PQ.fit(
        x = X_batch,
        y = y_batch,
        learning_rate = learning_rate,
        epochs = epochs
    )
    mean_loss = loss.sum().data[0] / batch_size
    self._update_Q_net()
    return mean_loss
```

学习方法：
```Python
def learning(self, gamma=0.99,
                   learning_rate=1e-5,
                   max_episodes=1000,
                   batch_size=64,
                   min_epsilon=0.2,
                   epsilon_factor=0.1,
                   epochs=1):
    '''learning的主要工作是构建经历，当构建的经历足够时，同时启动基于经历的学习
    '''
    total_steps, step_in_episode, num_episode = 0, 0, 0
    target_episode = max_episodes * epsilon_factor
    while num_episode < max_episodes:
        epsilon = self._decayed_epsilon(
            cur_episode = num_episode,
            min_epsilon = min_epsilon,
            max_epsilon = 1,
            target_episode = target_episode
        )
        self.state = self.env.reset()
        step_in_episode = 0
        loss, mean_loss = 0.00, 0.00
        is_done = False
        while not is_done:
            s0 = self.state
            a0 = self.performPolicy(s0, epsilon)
            s1, r1, is_done, info, total_reward = self.act(a0)
            step_in_episode += 1

            if self.total_trans > batch_size:
                loss += self._learn_from_memory(
                    gamma,
                    batch_size,
                    learning_rate,
                    epochs
                )
        
        mean_loss = loss / step_in_episode
        print("{0} epsilon:{1:3.2f}, loss:{2:.3f}".\
            format(self.experience.last, epsilon, mean))

        total_steps += step_in_episode
        num_episode += 1
    return
```
添加一些重要的辅助方法
```Python
def _decayed_epsilon(self,cur_episode,
                          min_epsilon,
                          max_epsilon,
                          target_episode):
    '''获得一个在一定范围内的epsilon
    '''
    slope = (min_epsilon - max_epsilon) / (target_episode)
    intercept = max_epsilon
    return max(min_epsilon, slope * cur_episode + intercept)

def _curPolicy(self, s, epsilon = None):
    '''依据更新策略的价值函数网络产生一个行为
    '''
    Q_s = self.PQ(s)
    rand_value = random()
    if epsilon is not None and rand_value < epsilon:
        return self.env.action_space.sample()
    else:
        return int(np.argmax(Q_s))

def performPolicy(self, s, epsilon=None):
    return self._curPolicy(s, epsilon)

```
最后，还需要一个方法来将一直在更新参数的近似函数网络的参数拷贝给评估网络：
```Python
def _update_Q_net(self):
    self.Q = self.PQ.clone()
```

**观察DQN在PuckWorld和CartPole环境中的表现**
测试代码：
```Python
from random import random, choice
import gym
from core import Transition, Experience, Agent
from approximator import Approximator
from agents import ApproxQAgent
import torch

def testApproxQAgent():
    env.gym.make()

    agent = ApproxQAgent(env,
                         trans_capacity = 10000, # 记忆容量
                         hidden_dim = 16         # 隐藏层神经元的数量
                        )
    env.reset()
    print('Learning...')
    agent.learning(
        gamma = 0.99,               # 衰减因子
        learning_rate = 1e-3,       # 学习率
        batch_size = 64,            # 批处理的规模
        max_episode = 20000,        # 最大训练量
        min_epsilon = 0.01,         # 最小 epsilon
        epsilon_factor = 0.3,       # 
        epochs = 2                  # 每个batch_size训练的次数
    )
if __name__ == "__main__":
    testApproxQAgent()
```

# 参考文献
[1] David Silver, reinforcement learning lecture 6 and 7
[2] 叶强, David Silver强化学习公开课中文讲解及实践, 知乎专栏
[3] Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. Nature, 2015, 518(7540): 529-533.