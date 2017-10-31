---
title: 【强化学习】算法实践-GridWorld Sarsa Sarsa($\lambda$)
date: 2017-10-28 15:52:36
tags:
    - RL
    - robotics
    - python
categories: 【强化学习】
---
{% img [GW] http://on99gq8w5.bkt.clouddn.com/GW.jpg?imageMogr2/thumbnail/600x600 %}
<!--more-->

# 0 引言
David Silver课程中使用了很多GridWorld的例子，如Samll GridWorld, Random Walk, Windy GridWorld, GridWorld, Cliff Walking. 这些例子中状态空间和动作空间都是离散的有限值，可以用gym中Discrete类来描述，另外这些例子都是用格子来表示世界，动作都是上下左右，所以可以考虑建立一个通用的GridWorld环境类，通过配置格子的数量、属性等来具体描述每一个例子。

# 1 通用的GridWorld环境类
整体思路是编写GridWorldEnv类，然后按照【强化学习】算法实践-OpenAI gym中介绍的办法来注册。

设计三个类 Grid GridMatrix GridWorldEnv
Grid 负责管理单个格子的属性，包括

# 2 Sarsa实现
通过设计一个Agent类的方式来实现。要怎么设计呢？

```Python
Class Agent(object):
    def __init__(self,env:Env):
        self.env = env     # agent所依附的环境
        self.Q = {}        # agent维护的Q表
        self.state = None  # agent当前的状态，可以写成观测，然后在后面加一个_obs_to_state的方法
    
    def performPolcy(self, state):                   # 执行一个策略
        pass

    def act(self, action):                           # 执行一个行为
        return self.env.step(aciotn)

    def learning(self, gamma, alpha, episode_num):   # 学习一个策略
        pass
```

Sarsa算法需要维护一个动作-状态值函数表$Q(x,a)$，也就是处于状态$x$的时候，采取动作$a$的价值。首先需要设计$Q$表的数据结构，怎么设计呢？字典套字典，即字典里的每一个健对应于状态名，其值对应于另一个新字典，这个新字典的键值是行为名，值对应相应的行为价值。

{"s0":{"a0":q_value00, "a1": q_value01, ...}, "s1":{"a0":q_value10, "a1": q_value11, ...}, ...}

为了对$Q$表进行查询、更新等操作，需要定义几个私有方法来供agent类内部使用：
```Python
# 判断状态s的Q值是否存在
def _is_state_in_Q(self, s):
    return self.Q.get(s) is not None

# 初始化某状态s的Q值
def _init_state_value(self, s_name, randomized=True):
    if not self._is_state_in_Q(s_name):
        self.Q[s_name] = {}
        for action in range(self.env.action_space.n):
            default_v = random() / 10 if randomized is True else 0.0
            self.Q[s_name][action] = default_v

# 确保某状态s的Q值存在
def _assert_state_in_Q(self, s, randomized=True):
    if not self._is_state_in_Q(s):
        self._init_state_value(s, randomized)

# 获取Q(s, a)
def _get_Q(self, s, a):
    self._assert_state_in_Q(s, randomized=True)
    return self.Q[s][a]

# 设置Q(s, a)
def _set_Q(self, s, a, value):
    self._assert_state_in_Q(s, randomized=True)
    self.Q[s][a] = value
```
下面就是Agent类的重头戏，performPolicy和learning

```Python
def performPolicy(self, s, episode_num, use_epsilon):
    epsilon = 1.00 / (episode_num + 1)
    Q_s = self.Q[s]
    str_act = "unkown"
    rand_value = random()
    action = None
    if use_epsilon and rand_value < epsilon:
        action = self.env.action_space.sample()
    else:
        str_act = max(Q_s, key=Q_s.get)
        action = int(str_act)
```
执行策略中的use_epsilon参数可以使我们随时切换是否使用$\epsilon$。通过这样的设置，今后可以很容易将SARSA算法修改为Q学习算法。

learning部分Sarsa算法的实现：(只向前看一步)
```Python
def learning(self, gamma, alpha, max_episode_num):
    total_time, time_in_episode, num_episode = 0, 0, 0
    while num_episode < max_episode_num:     # 设置终止条件
        self.state = self.env.reset()        # 环境初始化
        s = self._get_state_name(self.state) # 获取个体对于观测的命名
        self.env.render()                    # 显示UI界面
        a = self.performPolicy(s, num_episode, use_epsilon=True)

        time_in_episode = 0
        is_done = False
        while not is_done:
            s_prime, r, is_done, info = self.act(a) # 执行行为
            self.env.render()                       # 更新UI界面
            s_prime = self._get_state_name(s_prime) # 获取新的名字
            self._assert_state_in_Q(s_prime, randomized = True)

            a_prime = self.performPolicy(s_prime, num_episode, use_epsilon=True) # 获取a'
            
            old_q = self._get_Q(s, a)
            td_target = r + gamma * self._get_Q(s_prime, a_prime)
            new_q = old_q + alpha * (td_target - old_q)
            self._set_Q(s, a, new_q)

            # 终端显示最后的Episode的信息
            if num_episode == max_episode_num:
                print ("t:{0:>2}: s:{1}, a:{2:2}, s1:{3}".\
                    format(time_in_episode, s, a, s_prime))

            s, a = s_prime, a_prime
            time_in_episode += 1

        # 显示每一个Episode花费多少步
        print("t:{0:>2}: s:{1}, a:{2:2}, s1:{3}".\
            format(time_in_episode, s, a, s_prime))
        total_time += time_in_episode
        num_episode += 1
    return 
```
试验问题：会出现反复，结果好坏相间，不具有严格单调性

# 3 Sarsa($\lambda$)实现
Agent类的部分和Sarsa基本一致，主要区别在两点：一是learning方法，二是Sarsa($\lambda$)算法要维护一个E表(效用追踪)。
```Python
def learning(self, lambda_, gamma, alpha, max_episode_num):
    total_time = 0
    time_in_episode = 0
    num_episode = 1
    while num_episode <= max_episode_num:
        self._resetEValue()  # 要点一：每个episode都要置0
        s = self._name_state(self.env.reset())
        a = self.performPolicy(s, num_episode)
        self.env.render()

        time_in_episode = 0
        is_done = False
        while not is_done:
            s_prime, r, is_done, info = self.act(a)
            self.env.render()
            s_prime = self._name_state(s_prime)
            self._assert_state_in_QE(s_prime, randomized=True)

            a_prime = self.performPolicy(s_prime, num_episode)

            q = self._get_(self.Q, s, a)
            q_prime = self._get_(self.Q, s_prime, a_prime)
            delta = r + gamma * q_prime - q

            e = self._get_(self.E, s, a)
            e = e + 1
            self._set_(self.E, s, a, e)  # set E before update E

            # 要点二：在整个状态空间和行为空间上，对Q和E进行更新
            state_action_list = list(zip(self.E.keys(), self.E.values()))
            for s, a_es in state_action_list:
                for a in range(self.env.action_space.n):
                    e_value = a_es[a]
                    old_q = self._get_(self.Q, s, a)
                    new_q = old_q + alpha * delta * e_value
                    new_e = gamma * lambda_ * e_value
                    self._set_(self.Q, s, a, new_q)
                    self._set_(self.E, s, a, new_e)

            if num_episode == max_episode_num:
                # print current action series
                print("t:{0:>2}: s:{1}, a:{2:10}, s1:{3}".\
                      format(time_in_episode, s, a, s_prime))

            s, a = s_prime, a_prime
            time_in_episode += 1

        print("Episode {0} takes {1} steps.".\
                format(num_episode, time_in_episode))
        total_time += time_in_episode
        num_episode += 1
    return
```
PS: 容易在代码的世界里迷失，找不到的公式的感觉

# 4 理想与现实的差距-格子世界毕竟是格子世界

# 参考文献
[1] David Silver, reinforcement learning lecture
[2] 叶强, David Silver强化学习公开课中文讲解及实践, 知乎专栏
