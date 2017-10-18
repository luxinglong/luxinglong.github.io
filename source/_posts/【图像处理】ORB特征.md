---
title: 【图像处理】ORB特征
date: 2017-04-17 20:34:58
tags: 
    - image processing
    - local feature
categories: 【图像处理】
---

# 0 学习目标
* 理解ORB特征的原理和优缺点
* 利用OpenCV实现ORB特征
* 了解ORB特征在ORB-SLAM中的应用

# 1 原理
ORB (Oriented FAST and Rotated BRIEF)特征基于BRIEF特征，但是具有旋转不变性，并具有抗干扰的能力。ORB特征的贡献有：
* 为FAST角点增加了可以快速计算和精确的方向信息
* 可以快速计算的具有方向信息的BRIEF描述子rBRIEF
* 对rBRIEF方差和相关特性进行了分析
* 一种基于学习的具有旋转不变性的解相关BRIEF特征，在最近邻算法应用中表现很好

<!--more-->

## oFAST:FAST Keyoint Orientation
ORB特征点的提取是在FAST的基础上改进的，称为oFAST，也就是为每个FAST特征点增加一个方向信息，以此来使其具有旋转不变性。

**FAST特征检测**
1. 首先要检测FAST-9角点；
2. 在每个FAST角点处，计算Harris角点响应值，根据响应值进行排序，选取前N个角点；
3. 计算图像的高斯金字塔，在不同的尺度上进行上述两步。
这样便可以使ORB特征具有**尺度不变性**。

**如何使其具有旋转不变性：**
回顾一下BRIEF描述子的计算过程：
在当前关键点P周围以一定模式选取N个点对，组合这N个点对的$\tau$操作的结果就为最终的描述子。当我们选取点对的时候，是以当前关键点为原点，以水平方向为X轴，以垂直方向为Y轴建立坐标系。当图片发生旋转时，坐标系不变，同样的取点模式取出来的点却不一样，计算得到的描述子也不一样，这是不符合我们要求的。因此我们需要重新建立坐标系，使新的坐标系可以跟随图片的旋转而旋转。这样我们以相同的取点模式取出来的点将具有一致性。
打个比方，我有一个印章，上面刻着一些直线。用这个印章在一张图片上盖一个章子。印章不变动的情况下，转动下图片，再盖一个章子，但这次取出来的点对就和之前的不一样。为了使2次取出来的点一样，我需要将章子也旋转同一个角度再盖章。ORB在计算BRIEF描述子时建立的坐标系是以关键点为圆心，以关键点和取点区域的形心的连线为X轴建立2维坐标系。
{% img [orientation] http://on99gq8w5.bkt.clouddn.com/orientation.png %}
P为关键点。圆内为取点区域，每个小格子代表一个像素。现在我们把这块圆心区域看做一块木板，木板上每个点的质量等于其对应的像素值。根据积分学的知识我们可以求出这个密度不均匀木板的质心Q。计算公式如下。其中R为圆的半径。
$$
m_{pq}=\sum_{x=-R}^R\sum_{y=-R}^Rx^py^qI(x,y)
$$
$$
C=(m_{10}/m_{00},m_{01}/m_{00})
$$
$$
\theta=atan2(m_{01},m_{10});
$$
atan2表示反正切，得到的$\theta$值就是FAST特征点的主方向。

我们知道圆心是固定的而且随着物体的旋转而旋转。当我们以PQ作为坐标轴时，在不同的旋转角度下，我们以同一取点模式取出来的点是一致的。这就解决了旋转一致性的问题。

另外，文献[1]还比较了灰度重心法(Intensity Centroid)和基于梯度的MAX、BIN方法在旋转不变性上的表现。MAX方法选择X、Y方向最大的梯度方向，BIN方法在候选点周围每隔10度，计算一次梯度，选择最大的梯度方向。基于梯度的方法都要提前对图像进行平滑处理。


## rBRIEF:Rotation-Aware BRIEF

**如何解决对噪声敏感的问题：**
BRIEF使用的是pixel跟pixel的大小来构造描述子的每一个bit。这样的后果就是对噪声敏感。因此，在ORB的方案中，做了这样的改进，不再使用pixel-pair，而是使用patch-pair，也就是说，在$31\times 31$的窗口中，产生随机点对，然后以随机点为中心，取一个$5\times 5$的patch,对patch的像素值之和进行比较。（可以通过积分图快速计算）。

**使BRIEF特征具有旋转不变性**
得到特征点的主方向$\theta$，BRIEF作者提出将每个patch旋转，然后计算BRIEF特征，但是这种方法计算效率不高。ORB采用的做法是：在每个特征点处，对产生的所有点对，进行旋转，然后进行判别，生成二进制串。
首先：对于候选点区域的n个点对，建立一个$2\times n$的矩阵
$$
S=\left[
\begin{matrix}
x_1 & \cdots &x_n\\
y_1 & \cdots &y_n\\
\end{matrix}    
\right]
$$
利用主方向$\theta$建立的旋转矩阵$R_{\theta}$，对S进行旋转变换得到$S_{\theta}$
$$
S_{\theta}=R_{\theta}S
$$
这样就可以计算BRIEF特征了
$$
g_n(p,\theta):=f_n(p)|(x_i,y_i)\in S_{\theta}=\sum_{1\le i \le n}2^{i-1}\tau(p;x_i,y_i)
$$

为了提高计算效率，ORB将360度离散成步长为$2\pi/30$的30个点，然后计算响应的$S_{theta}$，当主方向$\theta$计算出来，查找相应表就可以得到$S_{theta}$，因为整张图像的旋转角度是一致的，因此$S_{theta}$可以用来计算BRIEF特征。
## 优点
1. 具有旋转不变性
2. 具有尺度不变性
3. 性能与SIFT接近，但是计算比SIFT快2的阶数次方

## 缺点
1. 不具有光照不变性

# 2 源代码解析
{% codeblock orb.cpp %}
// 总结：计算Harris角点响应值
// 输入：
// 返回：
static void
HarrisResponses(const Mat& img, const std::vector<Rect>& layerinfo,
                std::vector<KeyPoint>& pts, int blockSize, float harris_k)
{
    CV_Assert( img.type() == CV_8UC1 && blockSize*blockSize <= 2048 );

    size_t ptidx, ptsize = pts.size();

    const uchar* ptr00 = img.ptr<uchar>();
    int step = (int)(img.step/img.elemSize1());
    int r = blockSize/2;

    float scale = 1.f/((1 << 2) * blockSize * 255.f);
    float scale_sq_sq = scale * scale * scale * scale;

    AutoBuffer<int> ofsbuf(blockSize*blockSize);
    int* ofs = ofsbuf;
    for( int i = 0; i < blockSize; i++ )
        for( int j = 0; j < blockSize; j++ )
            ofs[i*blockSize + j] = (int)(i*step + j);

    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        int x0 = cvRound(pts[ptidx].pt.x);
        int y0 = cvRound(pts[ptidx].pt.y);
        int z = pts[ptidx].octave;

        const uchar* ptr0 = ptr00 + (y0 - r + layerinfo[z].y)*step + x0 - r + layerinfo[z].x;
        int a = 0, b = 0, c = 0;

        for( int k = 0; k < blockSize*blockSize; k++ )
        {
            const uchar* ptr = ptr0 + ofs[k];
            int Ix = (ptr[1] - ptr[-1])*2 + (ptr[-step+1] - ptr[-step-1]) + (ptr[step+1] - ptr[step-1]);
            int Iy = (ptr[step] - ptr[-step])*2 + (ptr[step-1] - ptr[-step-1]) + (ptr[step+1] - ptr[-step+1]);
            a += Ix*Ix;
            b += Iy*Iy;
            c += Ix*Iy;
        }
        pts[ptidx].response = ((float)a * b - (float)c * c -
                               harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;
    }
}

// 总结：通过灰度重心计算特征点的主方向
// 输入：
// 返回

static void ICAngles(const Mat& img, const std::vector<Rect>& layerinfo,
                     std::vector<KeyPoint>& pts, const std::vector<int> & u_max, int half_k)
{
    int step = (int)img.step1();
    size_t ptidx, ptsize = pts.size();

    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        const Rect& layer = layerinfo[pts[ptidx].octave];
        const uchar* center = &img.at<uchar>(cvRound(pts[ptidx].pt.y) + layer.y, cvRound(pts[ptidx].pt.x) + layer.x);

        int m_01 = 0, m_10 = 0;

        // Treat the center line differently, v=0
        for (int u = -half_k; u <= half_k; ++u)
            m_10 += u * center[u];

        // Go line by line in the circular patch
        for (int v = 1; v <= half_k; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = u_max[v];
            for (int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v*step], val_minus = center[u - v*step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        pts[ptidx].angle = fastAtan2((float)m_01, (float)m_10);
    }
}

// 总结：计算BRIEF特征
// 输入：
// 返

static void
computeOrbDescriptors( const Mat& imagePyramid, const std::vector<Rect>& layerInfo,
                       const std::vector<float>& layerScale, std::vector<KeyPoint>& keypoints,
                       Mat& descriptors, const std::vector<Point>& _pattern, int dsize, int wta_k )
{
    int step = (int)imagePyramid.step;
    int j, i, nkeypoints = (int)keypoints.size();

    for( j = 0; j < nkeypoints; j++ )  // 遍历所有的特征点
    {
        const KeyPoint& kpt = keypoints[j];
        const Rect& layer = layerInfo[kpt.octave];
        float scale = 1.f/layerScale[kpt.octave];
        float angle = kpt.angle;

        angle *= (float)(CV_PI/180.f);
        float a = (float)cos(angle), b = (float)sin(angle);

        const uchar* center = &imagePyramid.at<uchar>(cvRound(kpt.pt.y*scale) + layer.y,
                                                      cvRound(kpt.pt.x*scale) + layer.x);
        float x, y;
        int ix, iy;
        const Point* pattern = &_pattern[0];
        uchar* desc = descriptors.ptr<uchar>(j);

    #if 1
        #define GET_VALUE(idx) \
               (x = pattern[idx].x*a - pattern[idx].y*b, \
                y = pattern[idx].x*b + pattern[idx].y*a, \
                ix = cvRound(x), \
                iy = cvRound(y), \
                *(center + iy*step + ix) )
    #else
        #define GET_VALUE(idx) \
            (x = pattern[idx].x*a - pattern[idx].y*b, \
            y = pattern[idx].x*b + pattern[idx].y*a, \
            ix = cvFloor(x), iy = cvFloor(y), \
            x -= ix, y -= iy, \
            cvRound(center[iy*step + ix]*(1-x)*(1-y) + center[(iy+1)*step + ix]*(1-x)*y + \
                    center[iy*step + ix+1]*x*(1-y) + center[(iy+1)*step + ix+1]*x*y))
    #endif

        if( wta_k == 2 )
        {
            for (i = 0; i < dsize; ++i, pattern += 16)
            {
                int t0, t1, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                val = t0 < t1;
                t0 = GET_VALUE(2); t1 = GET_VALUE(3);
                val |= (t0 < t1) << 1;
                t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                val |= (t0 < t1) << 2;
                t0 = GET_VALUE(6); t1 = GET_VALUE(7);
                val |= (t0 < t1) << 3;
                t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                val |= (t0 < t1) << 4;
                t0 = GET_VALUE(10); t1 = GET_VALUE(11);
                val |= (t0 < t1) << 5;
                t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                val |= (t0 < t1) << 6;
                t0 = GET_VALUE(14); t1 = GET_VALUE(15);
                val |= (t0 < t1) << 7;

                desc[i] = (uchar)val;
            }
        }
        else if( wta_k == 3 )
        {
            for (i = 0; i < dsize; ++i, pattern += 12)
            {
                int t0, t1, t2, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1); t2 = GET_VALUE(2);
                val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

                t0 = GET_VALUE(3); t1 = GET_VALUE(4); t2 = GET_VALUE(5);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

                t0 = GET_VALUE(6); t1 = GET_VALUE(7); t2 = GET_VALUE(8);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

                t0 = GET_VALUE(9); t1 = GET_VALUE(10); t2 = GET_VALUE(11);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;

                desc[i] = (uchar)val;
            }
        }
        else if( wta_k == 4 )
        {
            for (i = 0; i < dsize; ++i, pattern += 16)
            {
                int t0, t1, t2, t3, u, v, k, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                t2 = GET_VALUE(2); t3 = GET_VALUE(3);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val = k;

                t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                t2 = GET_VALUE(6); t3 = GET_VALUE(7);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 2;

                t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                t2 = GET_VALUE(10); t3 = GET_VALUE(11);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 4;

                t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                t2 = GET_VALUE(14); t3 = GET_VALUE(15);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 6;

                desc[i] = (uchar)val;
            }
        }
        else
            CV_Error( Error::StsBadSize, "Wrong wta_k. It can be only 2, 3 or 4." );
        #undef GET_VALUE
    }
}


{% endcodeblock %}
# 3 OpenCV实现

# 4 参考文献
[1] Rublee E, Rabaud V, Konolige K, et al. ORB: An efficient alternative to SIFT or SURF[C]//Computer Vision (ICCV), 2011 IEEE International Conference on. IEEE, 2011: 2564-2571.
[2] OpenCV documentation, http://docs.opencv.org/2.4/index.html