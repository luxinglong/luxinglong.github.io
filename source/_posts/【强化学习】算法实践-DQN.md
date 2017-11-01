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

# 2 


<video src='http://on99gq8w5.bkt.clouddn.com/Decentralized%20Multi-agent%20Collision%20Avoidance%20with%20Deep%20Reinforcement%20Learning.mp4' type='video/mp4' controls='controls'  width='100%' height='100%'>
</video>
# 参考文献
[1] David Silver, reinforcement learning lecture 6
[2] 
