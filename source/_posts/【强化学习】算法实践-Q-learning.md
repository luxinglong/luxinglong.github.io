---
title: 【强化学习】算法实践-Q-learning
date: 2017-11-06 15:40:01
tags:
    - RL
    - robotics
categories: 【强化学习】
---
{% img [q-learning] http://on99gq8w5.bkt.clouddn.com/q-learning.png?imageMogr2/thumbnail/600x600 %}
<!--more-->
# 0 引言
从封面的伪代码可以看到Q-leading算法的实现细节。要理解Q-learning算法，首先要理解什么是同策略，什么是异策略？

我对同异策略经过了两个阶段：
1. 同策略就是评估和提升的是同一个策略，异策略就是评估和提升的不是同一个策略；
2. 同策略就是产生学习数据的策略和提升的策略是同一个策略，异策略就是产生数据和提升的策略是同一个策略。

可以通过比较sarsa(同策略)和q-learning(异策略)算法来总结两者的不同，下图是sarsa算法的伪代码：

{% img [sarsa-learning] http://on99gq8w5.bkt.clouddn.com/sarsa-learning.png?imageMogr2/thumbnail/600x600 %}

可以看到，sarsa算法需要同样的策略产生$A^{\prime}$，也就是“实践派”，利用亲自尝试的结果来作为target，必须是在线学习；而q-learning算法使用的是估计最好的动作，是“想象派“，可以离线学习，也就是可以把走过的经验或者别人的经验拿来，自己学习。

# 1 实现代码
q-learning的代码实现$^{[2]}$：
```Python
import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        # rows代表状态，cols代表动作
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
```

# 参考文献
[1] Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction.
[2] Morvan 强化学习教程 Q-learning
