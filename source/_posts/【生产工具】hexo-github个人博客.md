---
title: 【生产工具】hexo+github个人博客
date: 2017-10-19 00:27:05
tags:
    - tools
categories: 【生产工具】
---
# 0 前言
为了避免重复工作，整理积累所学的知识，特此搭建此博客(PS: 有装X的嫌疑)。

<!--more-->

# 1 搭建流程

1. 创建仓库，http://username.github.io；
2. 创建两个分支：master 与 hexo；
3. 设置hexo为默认分支（因为我们只需要手动管理这个分支上的Hexo网站文件）；
4. 使用git clone git@github.com:username/username.github.io.git拷贝仓库；
5. 在本地http://username.github.io文件夹下通过Git bash依次执行npm install hexo、hexo init、npm install 和 npm install hexo-deployer-git（此时当前分支应显示为hexo）;
6. 修改_config.yml中的deploy参数，分支应为master；
7. 依次执行git add .、git commit -m "..."、git push origin hexo提交网站相关的文件；
8. 执行hexo g -d生成网站并部署到GitHub上。
这样一来，在GitHub上的http://username.github.io仓库就有两个分支，一个hexo分支用来存放网站的原始文件，一个master分支用来存放生成的静态网页。

# 2 博客配置
博客的配置文件是_config.yml

# 3 文章发布
新建文章
hexo new "文章题目" ： 在source/_posts下便可以看到对应的文件夹和.md文件

本地部署
hexo server : 打开弹出的链接，便可以看到本地部署的效果

网络部署，如果要让别人看到你的博客，需要进行如下两步
hexo generate
hexo deploy 

也可以：执行hexo g -d生成网站并部署到GitHub上
# 4 多设备同步
创建两个分支 master hexo
设置hexo为git的默认分支，
设置master为hexo的发布的默认分支，修改_config.yml中的deploy参数，分支应为master
hexo用来存放网站的原始文件，master用来存放生成的静态网页

在本地对博客进行修改（添加新博文、修改样式等等）后，通过下面的流程进行管理。
1. 依次执行git add .、git commit -m "..."、git push origin hexo指令将改动推送到GitHub（此时当前分支应为hexo）；
2. 然后才执行hexo g -d发布网站到master分支上。

虽然两个过程顺序调转一般不会有问题，不过逻辑上这样的顺序是绝对没问题的（例如突然死机要重装了，悲催....的情况，调转顺序就有问题了）。

当重装电脑之后，或者想在其他电脑上修改博客，可以使用下列步骤：
1. 使用git clone git@github.com:CrazyMilk/CrazyMilk.github.io.git拷贝仓库（默认分支为hexo）；
2. 在本地新拷贝的http://CrazyMilk.github.io文件夹下通过Git bash依次执行下列指令：npm install hexo、npm install、npm install hexo-deployer-git（记得，不需要hexo init这条指令）。

# 5 博客多媒体
1. 公式
2. 图片
```JavaScript
{% img [img_name] "src"+?imageMogr2/thumbnail/600x600 %}
```
3. 视频
```JavaScript
<video src=' ' type='video/mp4' controls='controls'  width='100%' height='100%'>
</video>
```
 其中，src是七牛云的视频外链。

4. 代码段
```Markdown
    ```Language names
    code
    ``` # end of codeblock
```
Language names: C++\Bash\Python\JavaScript\Markdown


