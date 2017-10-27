---
title: 【强化学习】算法实践-Small GridWorld
date: 2017-10-26 20:57:32
tags:
    - RL
    - robotics
    - python
categories: 【强化学习】
---
# 0 引言
Small GridWorld问题是David Silver在强化学习课程中举的一个例子，该例子很好的描述了有模型的策略评估和策略改进方法。

{% img [gw] http://on99gq8w5.bkt.clouddn.com/gw.png?imageMogr2/thumbnail/500x400 %}

状态空间：16个位置，其中0和15代表终止状态
动作空间：上(n)、下(s)、左(w)、右(e)
状态转移：离开格子的动作，保持原状态；其他动作，100%转移到相应的状态。
奖励函数：终止状态奖励为0，其他状态奖励为-1
折扣因子：$\gamma=1$

当前策略：$\pi(n\mid \cdot)=\pi(s\mid \cdot)=\pi(w\mid \cdot)=\pi(e\mid \cdot)=\frac{1}{4}$

<!--more-->
# 1 值迭代实现
注：以下代码来自[1]
```Python
states = [i for i in range(16)]  # 定义状态空间
actions = ["n", "s", "w", "e"]   # 定义动作空间
gamma = 1.00                     # 定义折扣因子

values = [0  for _ in range(16)] # 初始化状态值函数
ds_actions = {"n": -4, "e": 1, "s": 4, "w": -1} # 建立动作与状态空间的联系

# 根据当前状态和采取的动作计算下一个状态
def nextState(s, a):
  next_state = s
  if (s%4 == 0 and a == "w") or (s<4 and a == "n") or \
     ((s+1)%4 == 0 and a == "e") or (s > 11 and a == "s"):
    pass
  else:
    ds = ds_actions[a]
    next_state = s + ds
  return next_state

# 获得当前状态的奖励值
def rewardOf(s):
  return 0 if s in [0,15] else -1

# 终止状态判断
def isTerminateState(s):
  return s in [0,15]

# 获得当前状态所有的后继状态，用列表返回
def getSuccessors(s):
  successors = []
  if isTerminateState(s):
    return successors
  for a in actions:
    next_state = nextState(s, a)
    # if s != next_state:
    successors.append(next_state)
  return successors

# 利用Bellman等式更新当前状态的值函数
def updateValue(s):
  sucessors = getSuccessors(s)
  newValue = 0  # values[s]
  num = 4       # len(successors)
  reward = rewardOf(s)
  for next_state in sucessors:
    newValue += 1.00/num * (reward + gamma * values[next_state])
  return newValue

# 执行一次值迭代，对所有状态进行更新
def performOneIteration():
  newValues = [0 for _ in range(16)]
  for s in states:
    newValues[s] = updateValue(s)
  global values
  values = newValues
  printValue(values)

# 将值函数按照4x4表格的形式打印出来
def printValue(v):
  for i in range(16):
    print('{0:>6.2f}'.format(v[i]),end = " ") # python3
    if (i+1)%4 == 0:
      print("")
  print()

# 定义主函数，迭代160次
def main():
  max_iterate_times = 160
  cur_iterate_times = 0
  while cur_iterate_times <= max_iterate_times:
    print("Iterate No.{0}".format(cur_iterate_times))
    performOneIteration()
    cur_iterate_times += 1
  printValue(values)

if __name__ == '__main__':
  main()
```
结果：
 0.00  -14.00 -20.00 -22.00 
-14.00 -18.00 -20.00 -20.00 
-20.00 -20.00 -18.00 -14.00 
-22.00 -20.00 -14.00   0.00 

问题：当迭代运行到第三步的时候，已经达到最优策略，但是还没有达到最优状态值函数。

# 参考文献
[1] 叶强, David Silver强化学习公开课中文讲解及实践, 知乎专栏
[2] David Silver, reinforcement learning lecture 2 and 3