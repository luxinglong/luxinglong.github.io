---
title: 【编程语言】Python-Matplotlib
date: 2017-11-29 19:41:35
tags:
    - Python
    - matplotlib
categories: 【编程语言】
---

{% img [matplotlib] http://on99gq8w5.bkt.clouddn.com/matplotlib.png?imageMogr2/thumbnail/400x400 %}
<!--more-->
# 0 引言
如果想用Python来取代Matlab，至少有一个库是必须要学习的，那就是matplotlib。数学建模的时候，C++出数据，但是出不了图的“窘困”还历历在目。为了一站式完成出数据和出图。痛下决心学习一下matplotlib。

# 1 基本功能
{% img [matplot_basic_demo] http://on99gq8w5.bkt.clouddn.com/matplot_basic_demo.png?imageMogr2/thumbnail/500x500 %}
```Python
import matplotlib.pyplot as plt
import numpy as np

# generate some data for x and y
x = np.linspace(-3,3,1000)

y = 2*x+1
y_ = x**2

plt.figure(num=2, figsize=(8,6))  # create figure and set number/size
l1 = plt.plot(x, y, label='linear') # plot
l2 = plt.plot(x, y_, color='red', linewidth=1.2, linestyle='--', label='non-linear')
plt.xlim(-1,2)  # axis x limitation
plt.ylim(-2,4)
plt.xlabel(r'$time:\ \tau$')  # label
plt.ylabel(r'$d_a$')
plt.title(r'$Bisc\ Figure\ Demo$') # title

plt.legend(labels=['best linear'], loc='lower right') # legend

plt.yticks([2],[r'$thresh$'])  # add tick for axis

# gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')  # remove right and top axis
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom') 
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))  # set bottom and left axis position
ax.spines['left'].set_position(('data', 0))

x0 = 1
y0 = 2*x0 + 1
plt.scatter(x0, y0, s=50, color='b')   # scatter a point 
plt.plot([x0,x0],[y0,0], 'k--', lw=2.5) # draw a line

# annotation method one
plt.annotate(r'$2x+1=%s$'%y0,xy=(x0,y0),xycoords='data',xytext=(+30,-30),textcoords='offset points', fontsize = 16, arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0.2'))

# method two
plt.text(-0.7,3, r'$This\ is\ the\ some\ text.\mu\ \sigma_i\ \alpha^2$',
        fontdict={'size':16, 'color':'r'})

# make axis index or tick more clear 
for label in ax.get_xticklabels()+ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7))

plt.show()

```

# 2 其他图像类型
## 2.1 Scatter
{% img [scatter] http://on99gq8w5.bkt.clouddn.com/scatter.png?imageMogr2/thumbnail/500x500 %}

```Python
import matplotlib.pyplot as plt
import numpy as np

n = 1024    # data size
X = np.random.normal(0, 1, n) # 每一个点的X值
Y = np.random.normal(0, 1, n) # 每一个点的Y值
T = np.arctan2(Y,X) # for color value

plt.scatter(X, Y, s=75, c=T, alpha=.5)

plt.xlim(-1.5, 1.5)
plt.xticks(())  # ignore xticks
plt.ylim(-1.5, 1.5)
plt.yticks(())  # ignore yticks

plt.show()
```
## 2.2 Bar
{% img [bar] http://on99gq8w5.bkt.clouddn.com/bar.png?imageMogr2/thumbnail/500x500 %}
```Python
import matplotlib.pyplot as plt
import numpy as np

n = 12
X = np.arange(n)
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x, y in zip(X, Y1):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')

for x, y in zip(X, Y2):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x + 0.4, -y - 0.05, '-%.2f' % y, ha='center', va='top')

plt.xlim(-.5, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())

plt.show()
```

## 2.3 Contours
{% img [contour] http://on99gq8w5.bkt.clouddn.com/contour.png?imageMogr2/thumbnail/500x500 %}
```Python
import matplotlib.pyplot as plt
import numpy as np

def f(x,y):
    # the height function
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X,Y = np.meshgrid(x, y)

# use plt.contourf to filling contours
# X, Y and value for (X,Y) point
plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)

# use plt.contour to add contour lines
C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)

plt.clabel(C, inline=True, fontsize=10)

plt.xticks(())
plt.yticks(())

plt.show()
```

## 2.4 Image
{% img [image] http://on99gq8w5.bkt.clouddn.com/image.png?imageMogr2/thumbnail/500x500 %}
```Python
import matplotlib.pyplot as plt
import numpy as np

a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)

plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')

plt.colorbar(shrink=.92)

plt.xticks(())
plt.yticks(())
plt.show()
```
## 2.5 3D
{% img [3d_figure] http://on99gq8w5.bkt.clouddn.com/3d_figure.png?imageMogr2/thumbnail/500x500 %}

文章开头那张图的感觉，有木有？

```Python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

# X, Y value
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)    # x-y 平面的网格
R = np.sqrt(X ** 2 + Y ** 2)
# height value
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
ax.contourf(X, Y, Z, zdir='x', offset=-4, cmap=plt.get_cmap('rainbow'))
ax.contourf(X, Y, Z, zdir='y', offset=4, cmap=plt.get_cmap('rainbow'))

plt.show()
```
# 3 多图合并与显示
{% img [multi-figure] http://on99gq8w5.bkt.clouddn.com/multi-figure.png?imageMogr2/thumbnail/500x500 %}
```Python
import matplotlib.pyplot as plt

plt.figure()

plt.subplot(2,2,1)
plt.plot([0,1],[0,1])
plt.subplot(2,2,2)
plt.plot([0,1],[0,2])
plt.subplot(223)
plt.plot([0,1],[0,3])
plt.subplot(224)
plt.plot([0,1],[0,4])

plt.show()  # 展示
```
## 更复杂的多图排列

## 图中图

## 次坐标轴

# 4 Animation 动画

# 参考文献
[1] 利用Python进行数据分析
[2] Morvan matplotlib lecture
[3] http://matplotlib.org/gallery.html
