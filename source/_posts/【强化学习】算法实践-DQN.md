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


# 3 DQN算法实现

# 参考文献
[1] David Silver, reinforcement learning lecture 6 and 7
[2] 叶强, David Silver强化学习公开课中文讲解及实践, 知乎专栏
