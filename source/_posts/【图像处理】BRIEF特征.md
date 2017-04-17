---
title: 【图像处理】BRIEF特征
date: 2017-04-17 13:42:27
tags: 
    - image processing
    - local feature
categories: 【图像处理】
---
# 0 学习目标
* 理解BRIEF特征
* 利用BRIEF特征，实现特征匹配

# 1 原理
特征点的descriptor是很多视觉任务的重点，例如在物体检测，三维重建，SLAM中，都会用到特征检测和匹配。在视觉SLAM这种对实时性要求很高的领域，descriptor的快速计算显得尤为关键。BRIEF(Binary Robust Independent Elementary Features)是一种很方便计算，而且很具有鲁棒性的特征描述形式。

我们知道SIFT特征是一种具有旋转、尺度、光照不变的特性，是一种非常理想的局部特征描述方式。但是它需要用一个128维的向量来进行表示，计算和匹配都非常耗时。SURF在计算量上对其进行了改进，使用64维向量来表示，但是每一位是一个浮点数，需要4个字节来保存，共需要256个字节保存，这对于存储的需要很大。为了减小它的计算量和存储量，大致上有三种方法：
<!--more-->
1. 采用PCA或者LDE进行降维，在保持特征质量的同时，减小了特征向量的维数。
2. 对每一位浮点数进行量化(Quantization)，使用较少的bits来保存量化后的浮点数。
3. 将浮点数向量转化成二值串，并使用Hamming距离来计算特征之间的相似度。

上述的三种方法，虽然可以得到一个简化板的特征向量，方便计算和存储。问题是这三种方法都需要预先提取一个长特征向量，然后再进行简化。而文献[1]直接提取简化的特征。
# Hamming distance
**汉明距离**是以理查德·卫斯里·汉明的名字命名的。在信息论中，两个等长字符串之间的汉明距离是两个字符串对应位置的不同字符的个数。换句话说，它就是将一个字符串变换成另外一个字符串所需要替换的字符个数。
**例如**
    The Hamming distance between 1011101 and 1001001 is 2. 
    The Hamming distance between 2143896 and 2233796 is 3. 
    The Hamming distance between "toned" and "roses" is 3. 

**汉明重量**是字符串相对于同样长度的零字符串的汉明距离，也就是说，它是字符串中非零的元素个数：对于二进制字符串来说，就是1的个数，所以 11101的汉明重量是4。

如果把a和b两个单词看作是向量空间中的元素，则它们之间的汉明距离等于它们汉明重量的差a-b。如果是二进制字符串a和b，汉明距离等于它们汉明重量的和a+b或者a和b汉明重量的异或a XOR b。汉明距离也等于一个n维的超立方体上两个顶点间的曼哈顿距离，n指的是单词的长度。

汉明距离可以在通信中累计定长二进制字中发生翻转的错误数据位，所以它也被称为信号距离。汉明重量分析在包括信息论、编码理论、密码学等领域都有应用。但是，如果要比较两个不同长度的字符串，不仅要进行替换，而且要进行插入与删除的运算，在这种场合下，通常使用更加复杂的编辑距离等算法。

## 具体方法
首先，以特征点为中心，定义一个$S\times S$的patch区域(在OpenCV中默认48)，然后在这个patch上定义函数$\tau$
$$
\begin{eqnarray}
\tau (p;x,y)=
\begin{cases}
1,& if p(x) < p(y)\\
0,& otherwise\\
\end{cases}
\end{eqnarray}
$$
其中，$p(x)$表示像素点$x=(u,v)^T$的像素值。
BRIEF特征可以定义为一个$n_d$维的二进制串：
$$
\begin{eqnarray}
f_{n_d}(p):=\sum_{1\le i \le n_d} 2^{i-1}\tau (p;x_i,y_i)
\end{eqnarray}
$$
其中，$n_d$可以取128，256，512，相应的特征表示为BRIEF-16，BRIEF-32和BRIEF-64.
具体实施过程中：
1. 为了增强BRIEF特征的抗干扰性和可重复性，需要选择合适的Gaussian Kernel进行平滑处理；在OpenCV中默认的kernel尺寸为9；
2. 在$S\times S$的patch中，选择$n_d$个测试点对$(x_i,y_i)$；文献[1]中介绍了五种方法：
    * X和Y都服从在[-S/2，S/2]范围内的均匀分布，且相互独立；
    * X和Y都服从均值为0，方差为$S^2/25$的高斯分布，且相互独立，即X和Y都以原点为中心，进行同方差的高斯分布；
    * X服从均值为0，方差为$S^2/25$的高斯分布，而Y服从均值为$x_i$，方差为$S^2/100$的高斯分布，即先确定X的高斯分布得到$x_i$，同方法2，然后以$x_i$为中心，进行高斯分布确定$y_i$；
    * 在引入了空间量化的不精确极坐标网格的离散位置内，随机采样，得到$x_i$和$y_i$；
    * $x_i$固定在原点处，$y_i$是所有可能的极坐标网格内的值。

    通过实验对比可知，前4种方法要好于第5种方法，而在前4种方法中，第2种方法会表现出少许的优势。

    在实际应用中，虽然点对都是按一定规则随机选择的，但在确定了补丁区域大小S的情况下，点对的坐标位置一旦随机选定，就不再更改，自始自终都用这些确定下来的点对坐标位置。也就是说这些点对的坐标位置其实是已知的，在编写程序的时候，这些坐标事先存储在系统中，在创建描述符时，只要调用这些坐标即可。另外，不但点对的坐标位置是确定好的，点对的两个像素之间的顺序和点对的顺序也必须是事先确定好的，这样才能保证描述符的一致性。点对的两个像素之间的顺序指的是在公式1中，两个像素哪个是$x_i$，哪个是$y_i$，因为在比较时是$x_i$的灰度值小于$y_i$的灰度值时，$\tau$才等于1。点对的顺序指的是$n_d$个点对之间要排序，这样二值位字符串中的各个位（公式2）就以该顺序排列。
3. 计算Hamming距离。

最后需要强调的是，BRIEF仅仅是一种特征点的描述符方法，它不提供特征点的检测方法。Calonder推荐使用CenSurE方法进行特征点的检测，该方法与BRIEF配合使用，效果会略好一些。在Opencv2.4.9中也提供了CenSurE方法，但是使用Star这个别名。

## 优点
直接提取二进制串，计算快，占用内存小。
## 缺点
1. 不具有旋转不变性
2. 对噪声敏感，需要选取合适的Gaussian Kernel进行平滑处理
3. 不具有尺度不变性

>后面介绍的ORB特征，主要解决问题1和2.

# 2 源代码解析
{% codeblock brief.cpp %}
/*
 * BRIEF Descriptor
 */
class BriefDescriptorExtractorImpl : public BriefDescriptorExtractor
{
public:
    enum { PATCH_SIZE = 48, KERNEL_SIZE = 9 };

    // bytes is a length of descriptor in bytes. It can be equal 16, 32 or 64 bytes.
    BriefDescriptorExtractorImpl( int bytes = 32, bool use_orientation = false );

    virtual void read( const FileNode& );
    virtual void write( FileStorage& ) const;

    virtual int descriptorSize() const;
    virtual int descriptorType() const;
    virtual int defaultNorm() const;

    virtual void compute(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors);

protected:
    typedef void(*PixelTestFn)(InputArray, const std::vector<KeyPoint>&, OutputArray, bool use_orientation );

    int bytes_;
    bool use_orientation_;
    PixelTestFn test_fn_;
};
// 构造函数
BriefDescriptorExtractorImpl::BriefDescriptorExtractorImpl(int bytes, bool use_orientation) :
    bytes_(bytes), test_fn_(NULL)
{
    use_orientation_ = use_orientation;

    switch (bytes)
    {
        case 16:                      // nd=128
            test_fn_ = pixelTests16;
            break;
        case 32:                      // nd=256
            test_fn_ = pixelTests32;
            break;
        case 64:                      // nd=512
            test_fn_ = pixelTests64;
            break;
        default:
            CV_Error(Error::StsBadArg, "bytes must be 16, 32, or 64");
    }
}

{% endcodeblock %}
# 3 OpenCV实现特征提取和匹配

{% codeblock BRIEFDescriptor.cpp %}

#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	if (argc!=3)
	{
		cout << "Please input three parameters." << endl;
		return -1;
	}
	Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2]);
	resize(img1,img1,Size(640,480));
	resize(img2,img2,Size(640,480));

	vector<KeyPoint> keypoints1, keypoints2;

	// use CenSurE to get keypoints
    Ptr<xfeatures2d::StarDetector> detector=xfeatures2d::StarDetector::create();
	detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);

	// use FAST to get keypoints
	int threshold=50;
	//FAST(img1,keypoints1,threshold,true);
	//FAST(img2,keypoints2,threshold,true);

	// use BRIEF descriptor to represent keypoints
    Mat descriptors1, descriptors2;
    Ptr<xfeatures2d::BriefDescriptorExtractor> brief=xfeatures2d::BriefDescriptorExtractor::create();
    brief->compute(img1, keypoints1, descriptors1);
    brief->compute(img2, keypoints2, descriptors2);

	BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1,descriptors2,matches);


	// draw the match result
    namedWindow("BRIEF_matches",CV_WINDOW_NORMAL);
    Mat img_matches;
    drawMatches(img1,keypoints1,img2,keypoints2,matches,img_matches);
    imshow("BRIEF_matches",img_matches);
	imwrite("BRIEF_match.jpg",img_matches);
    waitKey(0);

    return 0;
}

{% endcodeblock %}

{% img [BRIEF_match] http://on99gq8w5.bkt.clouddn.com/BRIEF_match.jpg %}

# 4 参考文献
[1] Calonder M, Lepetit V, Strecha C, et al. Brief: Binary robust independent elementary features[J]. Computer Vision–ECCV 2010, 2010: 778-792.
[2] OpenCV documentation, http://docs.opencv.org/2.4/index.html

