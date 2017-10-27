---
title: 【强化学习】算法实践-OpenAI gym
date: 2017-10-26 21:12:07
tags:
    - RL
    - robotics
    - OpenAI gym
categories: 【强化学习】
---

# 0 引言
OpenAI gym是当下非常热门的强化学习库，使用者只需要定义环境就可以测试自己的强化学习算法。
本文主要介绍如何在ubuntu 16.04上配置gym开发环境，gym的建模思想，以及自己动手搭建一个gym环境。

<!--more-->
# 1 gym环境配置
```Bash
git clone https://github.com/openai/gym
cd gym
pip install -e . # minimal install
# pip install -e .[all] # full install (requires cmake and recent version pip)
```
# 2 gym建模思想
强化学习主要有两个对象，环境和智能体。环境的属性包括环境状态、，环境的行为包括状态转移、奖励机制；智能体的属性包括依附的环境、观察和获得的即时奖励，智能体的行为包括根据已知策略产生一个动作、执行动作改变环境、获得环境的状态和即时奖励。

按照这样的思路，可以提炼出伪代码如下：
```Python
Class Env():
    self.states # 环境状态集合
    self.
```

## gym的建模思想
gym库的核心文件是/gym/gym/core.py，这里定义了两个最基本的类Env和Space。前者是所有环境类的基类，后者是所有空间类的基类。从Space基类衍生出几个常用的空间类，其中最主要的是Discrete类和Box类。通过其__init__方法的参数以及其它方法的实现可以看出前者对应于一维离散空间，后者对应于多维连续空间。它们既可以应用在行为空间中，也可以用来描述状态空间。例如，Small GridWorld例子中，共有4x4个状态，每个状态只需要用一个数字来描述，这样就可以使用Discrete(16)对象来描述。Box空间可以定义多维空间，每一个维度可以用一个最小值和最大值来约束。

# 3 倒立摆CartPole-V0
gym中的倒立摆模型如下：
{% img [cartpole] http://on99gq8w5.bkt.clouddn.com/cartpole.png?imageMogr2/thumbnail/500x400 %}
状态空间：$\{x,\dot{x},\theta,\dot{\theta}\}$
动作空间：{1,-1}
奖励函数：
状态转移：由倒立摆的数学模型导出（忽略摩擦系数）

{% img [cp] http://on99gq8w5.bkt.clouddn.com/cp.png?imageMogr2/thumbnail/500x400 %}

倒立摆的数学模型
$$
mL\ddot{x}cos(\theta)+(mL^2+J)\ddot{\theta}=mgLsin(\theta) \\
(M+m)\ddot{x}+mL\ddot{\theta}cos(\theta)=F+mL\dot{\theta}^2sin(\theta)-b\dot{x}
$$

下面对CartPole-V0的环境文件进行分析，文件位置./gym/gym/envs/classic_control/cartpole.py
```Python
"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

# CartPoleEnv为gym.Env继承来的派生类
class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    # 构造函数
    def __init__(self):
        self.gravity = 9.8   # 重力加速度
        self.masscart = 1.0  # 小车的质量
        self.masspole = 0.1  # 摆的质量
        self.total_mass = (self.masspole + self.masscart) # 小车和摆的总质量
        self.length = 0.5 # actually half the pole's length  # 摆的重心到转轴之间的长度=半摆长
        self.polemass_length = (self.masspole * self.length) # 摆的质量和半摆长的乘积
        self.force_mag = 10.0    # 外加的作用力
        self.tau = 0.02  # seconds between state updates  # 状态更新的时间间隔

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360  # 摆的摆幅阈值
        self.x_threshold = 2.4  # 小车的平移阈值

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)  # 动作空间为2维离散空间
        self.observation_space = spaces.Box(-high, high) # 观测空间？

        self._seed()
        self.viewer = None     # 设置
        self.state = None      # 设置当前的状态为None

        self.steps_beyond_done = None  # 设置当前的步数为None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # 物理引擎 输入：当前状态+动作 输出：下一步状态 即时奖励 是否终止 调试项
    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state  # 系统的当前状态
        force = self.force_mag if action==1 else -self.force_mag  # 动作转化成作用力
        costheta = math.cos(theta)  # cos值
        sintheta = math.sin(theta)  # sin值
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))   # 角加速度计算公式
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass  # 位移加速度计算公式

        # 状态更新
        x  = x + self.tau * x_dot                    # 位移递推公式
        x_dot = x_dot + self.tau * xacc              # 速度递推公式
        theta = theta + self.tau * theta_dot         # 角度递推公式
        theta_dot = theta_dot + self.tau * thetaacc  # 角速度递推公式
        self.state = (x,x_dot,theta,theta_dot)       # 状态更新

        # 终止状态判断：位移超出范围、摆角超出范围
        done =  x < -self.x_threshold \              
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0  # 如果没有进入终止状态，奖励为1
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        
        # 返回下一时刻状态、奖励、终止信号、调试信息（空）
        return np.array(self.state), reward, done, {}

    # 重新初始化函数：智能体从经验（episodes）中学习经验，每次尝试一个episode，尝试完后要从头开始，重新初始化。
    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))  # 用随机数填充状态向量
        self.steps_beyond_done = None    # 设置当前的步数为None
        return np.array(self.state)  # 返回初始化后的状态

    # 图像引擎：可视化环境和智能体的交互过程
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # 设置显示窗口的长和宽
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2  # 现实世界的宽度
        scale = screen_width/world_width  # 现实世界的每一个单位宽度代表多少像素
        carty = 100 # TOP OF CART  # 小车顶部距离窗口下端的像素距离
        polewidth = 10.0  # 摆杆的宽度
        polelen = scale * 1.0  # 将1.0的现实世界长度转化成像素长度
        cartwidth = 50.0   # 小车的宽度
        cartheight = 30.0  # 小车的高度

        if self.viewer is None:
            # 导入rendering模块，利用其中的画图函数绘制图形
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height) # 绘制600x400的窗口
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2  
            axleoffset =cartheight/4.0
            # 用几何图形画车，依次车的顶点：左下、左上、右上、右下
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform() # 设置旋转和平移变换
            cart.add_attr(self.carttrans)  # 给小车添加属性
            self.viewer.add_geom(cart)  # 将小车加入图像引擎中

            # 用几何图形画摆杆，依次车的顶点：左下、左上、右上、右下
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4) # 设置颜色属性
            self.poletrans = rendering.Transform(translation=(0, axleoffset)) # 设置平移属性
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole) # 将摆杆加入图像引擎中

            # 用圆表示摆杆和小车的连接节点
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)  # 将节点加入图像引擎中

            # 用直线表示小车左右滑动的轨道
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track) # 将滑轨加入图像引擎中

        if self.state is None: return None # 当前状态为None的话，返回

        # 将当前的状态转化成小车的位置
        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)  # 设置平移
        self.poletrans.set_rotation(-x[2]) # 设置旋转

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
```

# 4 动手搭建自己的环境


# 参考文献
[1] 叶强, David Silver强化学习公开课中文讲解及实践, 知乎专栏
[2] 天津包子馅儿, 强化学习实战, 知乎专栏