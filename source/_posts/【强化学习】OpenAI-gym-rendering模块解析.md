---
title: 【强化学习】OpenAI gym-rendering模块解析
date: 2017-11-14 10:12:59
tags:
    - OpenAI gym
    - Python
categories: 【强化学习】
---

# 0 引言
由于要使用rendering模块搭建自己的仿真环境，但是对于画图库不是很熟悉，没办法得心应手。所以在这里拿来rendering模块进行解析，以求更便捷地画出自己的环境。

<!--more-->

建议使用IPython导入rendering模块，然后试验各个函数。

# 1 源码解析
文件地址：gym/gym/envs/classic_control/rendering.py
```Python
"""
2D rendering framework
"""
from __future__ import division
import os
import six
import sys

if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

from gym.utils import reraise
from gym import error

try:
    import pyglet   # 原来用的是pyglet库
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    from pyglet.gl import *  # 用到了OpenGL
except ImportError as e:
    reraise(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

import math
import numpy as np

RAD2DEG = 57.29577951308232  # 弧度转角度

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

# 重要的Viewer类，参数包括窗口的宽、长、是否显示
# 可以将Viewer类当成几何体的容器，创建出的几何体放在Viewer上显示
class Viewer(object):
    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width    # 窗口的宽
        self.height = height  # 窗口的高
        self.window = pyglet.window.Window(width=width, height=height, display=display) # 建立窗口对象
        self.window.on_close = self.window_closed_by_user
        self.geoms = []          # 一直显示的，放在这里
        self.onetime_geoms = []  # 只显示一次的东西，放在这里
        self.transform = Transform()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):   # 关闭窗口
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    # 设置画布的坐标系，默认左下角为原点
    def set_bounds(self, left, right, bottom, top): # 设置窗口的边界，变换的极限
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform = Transform(
            translation=(-left*scalex, -bottom*scaley),
            scale=(scalex, scaley))

    # 每次都显示的几何体的添加方法
    def add_geom(self, geom):
        self.geoms.append(geom)

    # 只显示一次的几何体的添加方法
    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    # 渲染函数，每调用一次，图形界面刷新一次，怎么控制刷新频率？？？
    def render(self, return_rgb_array=False):
        glClearColor(1,1,1,1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr

    # Convenience
    # 为了方便，给Viewer类添加了画圆方法、画填充多边形方法、画非填充多边形方法、
    # 画线方法，得到的几何体都只显示一次
    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom


    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1,:,0:3]

# 为几何体添加颜色和线宽属性
def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])

# 几何体类
class Geom(object):
    def __init__(self):
        self._color=Color((0, 0, 0, 1.0))
        self.attrs = [self._color]
    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()
    def render1(self):
        raise NotImplementedError
    def add_attr(self, attr):            # 添加属性的方法
        self.attrs.append(attr)
    def set_color(self, r, g, b):        # 设置颜色的方法
        self._color.vec4 = (r, g, b, 1)

# 几何体属性类
class Attr(object):
    def enable(self):
        raise NotImplementedError
    def disable(self):
        pass

# 几何体的变换属性，包括：平移、旋转、缩放
class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1,1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)
    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0) # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)
    def disable(self):
        glPopMatrix()
    def set_translation(self, newx, newy):             # 设置x,y轴的平移量
        self.translation = (float(newx), float(newy))
    def set_rotation(self, new):                       # 设置旋转量，单位：弧度
        self.rotation = float(new)
    def set_scale(self, newx, newy):                   # 设置x,y轴的缩放量
        self.scale = (float(newx), float(newy))

# 颜色类，四个参数：r,g,b,1，只需要关注前三个参数，继承自几何体属性类
class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4
    def enable(self):
        glColor4f(*self.vec4)

# 线型类，继承自几何体属性类
class LineStyle(Attr):
    def __init__(self, style):
        self.style = style
    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)
    def disable(self):
        glDisable(GL_LINE_STIPPLE)

# 线宽类，继承自几何体属性类
class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke
    def enable(self):
        glLineWidth(self.stroke)

# 点类，继承自几何体类
class Point(Geom):
    def __init__(self):
        Geom.__init__(self)
    def render1(self):
        glBegin(GL_POINTS) # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()

# 多边形类，继承自几何体类 <- 内部填充的几何体
class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v
    def render1(self):
        if   len(self.v) == 4 : glBegin(GL_QUADS)
        elif len(self.v)  > 4 : glBegin(GL_POLYGON)
        else: glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()

# 画圆函数，参数：半径、分辨率、是否填充，默认圆心在(0,0)点处，可使用变换属性移动
def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2*math.pi*i / res
        points.append((math.cos(ang)*radius, math.sin(ang)*radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)

# 画多边形函数，参数：多边形顶点列表、是否填充
def make_polygon(v, filled=True):
    if filled: return FilledPolygon(v)
    else: return PolyLine(v, True)

# 画多边形函数，参数：多边形顶点，默认不填充
def make_polyline(v):
    return PolyLine(v, False)

# 画“胶囊”函数，参数：长轴、短轴
# 通过两个圆和一个矩形合成
def make_capsule(length, width):
    l, r, t, b = 0, length, width/2, -width/2
    box = make_polygon([(l,b), (l,t), (r,t), (r,b)])
    circ0 = make_circle(width/2)
    circ1 = make_circle(width/2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom

# 将简单的几何体合并成复杂几何体的类，继承自几何体类
class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]
    def render1(self):
        for g in self.gs:
            g.render()

# 多边形类，继承自几何体类 <- 线段连接的几何体
class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)
    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()
    def set_linewidth(self, x):
        self.linewidth.stroke = x

# 线段类，继承自几何体类
class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()

# 图像类，继承自几何体类
# 可以将外界的图像导入Viewer中
class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False
    def render1(self):
        self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)

```
# 2 源码总结
最重要的类：Viewer提供了画布的功能

两个基类：Geom和Attr，分别代表几何体和几何体的属性

Geom的派生类：  Point: 点类
              FilledPolygon：填充多边形类
              PolyLine：线段连接多边形
              Line：线段类
              Image：图像类
              Compound：用于合成复杂几何体的类

Attr的派生类： Transform：运动属性
             Color：颜色属性
             LineStyle：线型属性
             LineWidth：线宽属性

## 2.1 如何绘制几何体呢？
rendering模块提供了以下几种绘图方法：
    make_circle：画圆，只需要指定半径，圆心默认为画布原点
    make_polygon: 画多边形，需要给顶点列表和是否填充标志
    make_polyline：画非填充的多边形，只需要顶点列表
    make_capsule： 画一个胶囊形状的几何体

绘图还需要指定颜色：set_color(r,g,b)
线宽和线型等属性，需要使用：add_attr()

## 2.2 如何控制几何体移动呢？
通过为几何体设置Transform属性对象，指定平移、旋转和缩放，进行移动。
并使用add_attr()添加移动属性。

## 2.3 如何将绘制的几何体添加到画布上？
Viewer类提供了两个方法：
    add_geom(): 一直显示的几何体的添加方法
    add_onetime(): 只显示一次的几何体的添加方法

## 2.4 画布坐标系是怎么样的？
默认情况下，原点在左下角，向右为x轴，向上为y轴。
还可以自己指定：
    set_bounds(left, right, bottom, top)

## 2.5 如果库中没有自己想要的几何体，怎么办呢？
可以通过几何体组合成自己想要的几何体，并使用Compound类来合成。这样就可以把自己的几何体作为一个整体，直接使用Geom类设置颜色、添加属性的方法了。
