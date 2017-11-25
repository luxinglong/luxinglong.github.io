---
title: 【应用程序】MFC基础知识
date: 2017-11-20 14:56:27
tags:
    - MFC
categories: 【应用程序】
---
# 0 引言
一个项目需要做办公自动化，自动提取word文档中的内容并保存到数据库中，而且还可以根据需要，自动生成word文档。

# 1 MFC基础知识
Windows应用程序包括，单文档、多文档和基于对话框。

1. 建一个基于对话框的MFC，可以参考（ http://blog.csdn.net/hhhh63/article/details/7652696）
2. 添加类库 C:\Program Files\Microsoft Office\OFFICE11\msword.olb ; 
3. 注释掉新生成的.h文件里边有//#import "C:\\Program Files\\Microsoft Office\\OFFICE11\\MSWORD.OLB" no_namespace，并添加代码#include<afxdisp.h>且放在所有Include的最前面
4. 然后就可以用相应函数操作word了，我是添加了一个button，添加点击事件，操作word.


# 参考文献
[1] visual C++ 项目和解决方案的区别 https://www.cnblogs.com/roucheng/archive/2016/05/30/cppxiangmu.html
[2] vc++操作word http://www.cnitblog.com/lifw/articles/vcpp_officeword.html
[3] 使用VC++2008操作word2003生成报表 http://blog.sina.com.cn/s/blog_412535a20100jsc4.html
[4] VC编程操作word2010生成表格 http://blog.csdn.net/clever101/article/details/52185032
[5] VC操作word和excel文件，查询与读写[依赖office环境]

https://support.microsoft.com/zh-cn/help/196776/office-automation-using-visual-c


