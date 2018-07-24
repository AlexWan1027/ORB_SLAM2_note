/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{

// 由Tracking::MonocularInitialization()进入
// sigma = 1.0， iterations = 200
Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    mK = ReferenceFrame.mK.clone();

    mvKeys1 = ReferenceFrame.mvKeysUn;

    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;
}

// Computes in parallel a fundamental matrix（基础矩阵） and a homography（单应性矩阵），
// Selects a model and tries to recover the motion and the structure from motion
bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    mvKeys2 = CurrentFrame.mvKeysUn;

    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());
    mvbMatched1.resize(mvKeys1.size());
    
    // vMatches12为第一帧图像帧的关键点在第二帧图像帧匹配对应的关键点索引
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        // 大于0即为有匹配，并组织特征点对
        // mvMatches12类型为vector<pair<int,int>>
        if(vMatches12[i]>=0)
        {
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
            mvbMatched1[i]=true;  // 有匹配点对标志位置1
        }
        else
            mvbMatched1[i]=false;  // 无匹配点对，标志位置为0
    }
    
    // 匹配的点对数，N
    const int N = mvMatches12.size();

    // Indices for minimum set selection
    // 
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    // 创建长度为mMaxIterations（RANSAC最大迭代次数），值为vector<size_t>(8,0)的向量mvSets
    // vector<size_t>(8,0)创建了长度为8，值为0的向量
    // 在所有的匹配点对中
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

    // DBoW2中的函数，作用：用来设置rand()产生随机数时的随机数种子
    // 相当于srand(0)
    DUtils::Random::SeedRandOnce(0);

    // 在所有匹配特征点对中随机选择8对匹配特征点为一组，共选择mMaxIterations组
    // RANSAC
    for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
	    // 产生0到N-1的随机数
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
	    // idx表示第二帧关键帧对应的索引，由随机数随机产生
            int idx = vAvailableIndices[randi];
            
	    // 将8个匹配点对保存至mvSets中
            mvSets[it][j] = idx;
            
	    // back()函数返回当前vector最末一个元素的引用
	    // pop_back():移除vector中的最后一个元素.
	    // 也就是将最后一元素顶替当前被选中出的值，然后再将最后一个元素删除，相当于将当前选中的值删除避免重复选中
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;   // 
    float SH, SF;    // 单应性矩阵和基本矩阵对应的分数，用于选择模型使用
    cv::Mat H, F;   // 用于保存计算所得的单应矩阵和基础矩阵

    // 创建双线程同时计算H和F
    // 多线程对象threadH和threadF，后面是构造函数，传入的分别为可调用函数，及其可调用函数的参数。
    // 其中ref的作用是使参数按引用传递
    // 线程从可调用函数开始执行
    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

    // Wait until both threads have finished
    // 等待两条线程执行结束
    threadH.join();
    threadF.join();

    // Compute ratio of scores
    // 计算得分比例
    float RH = SH/(SH+SF);

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    // 如果分数大于0.4选择单应性矩阵模型
    // 从对应的矩阵中恢复出两帧图像对应的相机的位姿变化
    if(RH>0.40)
        return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
    else //if(pF_HF>0.6)
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    return false;
}


// 假设场景为平面，通过匹配点求取Homography矩阵,并得到模型对应的得分SH
void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    const int N = mvMatches12.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    // 将mvKeys1和mvKey2归一化到均值为0，一阶绝对矩为1
    // vPn1表示归一化后的关键点坐标，归一化变换矩阵分别为T1、T2
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    // score为SH，即单应矩阵对应的分数
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    // 保存了随机产生的8个点的归一化后的点
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        // 得到选择的8个匹配点对对应的归一化点
        for(size_t j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        // 计算单应性矩阵H
        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
	// 解除归一化：H=T_inv*H'*T
        H21i = T2inv*Hn*T1;
	// 单应性矩阵的逆
        H12i = H21i.inv();
        // 计算得到单应矩阵对应的分数以及匹配点是否为内点的标志位
        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

	// 如果分数比上一个分数大，那么将当前单应性矩阵和分数给要输出的对应变量
        if(currentScore>score)
        {
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

// 8点法用于计算基础计算，原理详见多视图几何一书
void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = vbMatchesInliers.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    // 同单应性矩阵运算
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

	// 解除归一化
        F21i = T2t*Fn*T1;

	// 计算得到基本矩阵对应的分数以及匹配点是否为内点的标志位
        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

	// 如果分数比上一个分数大，那么将当前基本矩阵和分数给要输出的对应变量
        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

// 使用DLT(direct linear tranform)求解该模型，原理详见：多视图几何中文版第53页
// |x'|     | h1 h2 h3 ||x|
// |y'| = a | h4 h5 h6 ||y|  简写: x' = a H x, a为一个尺度因子
// |1 |     | h7 h8 h9 ||1|
// x' = a H x 
// ---> (x') 叉乘 (H x)  = 0
// ---> Ah = 0
// A = | 0  0  0 -x -y -1 xy' yy' y'|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
//     |-x -y -1  0  0  0 xx' yx' x'|
// 本来只需要4组点，但orb使用了8个点，所以变成了超定方程，可以通过SVD求解Ah = 0， A'A最小特征值对应的特征向量即为解

// 输入的是第一帧和第二帧图像中归一化后的关键点坐标，计算得到单应矩阵H
cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    // 匹配点对数
    const int N = vP1.size();

    // 参数矩阵A
    cv::Mat A(2*N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    cv::Mat u,w,vt;

    // SVD分解得到
    // vt：transposed matrix of right singular values 9*9
    // u：calculated left singular vectors 2N*2N
    // w:calculated singular values 2N*9
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // 返回A'A最小特征值对应的特征向量（vt的最后一行）即为解，将它reshape为3*3的矩阵
    return vt.row(8).reshape(0, 3);
}

// 基本矩阵的运算跟单应性矩阵运算有些区别，采用了8点法
// 原理详见多视图几何中文版第191页，基本矩阵相关内容
// 根据x'Fx = 0 整理可得：Af = 0，其中A = [x'x x'y x' y'x y'y y' x y 1]
cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();
    
    // A矩阵维度：N*9
    cv::Mat A(N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;
    // SVD： A = u*w*vt
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    // 得到的矩阵vt最后一列向量即为f的最小二乘解，并将其转换为3*3的矩阵
    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    // 由于基本矩阵F需要满足奇异性的条件，所以需要采取强迫约束得到满足的基本矩阵F'
    // 计算的方式为：对基本矩阵F进行SVD分解得到：F = u*w*vt，然后将对角阵w中的最后一个元素置为0，得到w'
    // 那么新的F'=u*w'*vt即为满足奇异性条件的基本矩阵
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;
}

// 对单应性矩阵模型打分
// 详见
// * - Author's paper - IV. AUTOMATIC MAP INITIALIZATION （2）
// * - Multiple View Geometry in Computer Vision - symmetric transfer errors: 4.2.2 Geometric distance
// * - Multiple View Geometry in Computer Vision - model selection 4.7.1 RANSAC
float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    const int N = mvMatches12.size();

    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);
 
    // 用于保存第i对匹配点是否为内点
    vbMatchesInliers.resize(N);

    float score = 0;
    
    // 基于卡方检验计算出的距离阈值（假设测量有一个像素的偏差）
    const float th = 5.991;

    // 
    const float invSigmaSquare = 1.0/(sigma*sigma);

    // N对匹配点
    for(int i=0; i<N; i++)
    {
        bool bIn = true;

	// Kp1和kp2分别对应匹配关键点对
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2
	// 通过单应性矩阵将图像2中的点反投影回图像1中得到对应的位置u2in1和v2in1
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

	// 得到像素位置误差平方和
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);
        // 测量误差的标准方差，由于内点判定条件是：d^2 < t^2=th*Sigma^2，所以我们对距离误差平方除以Sigma^2
        const float chiSquare1 = squareDist1*invSigmaSquare;

	// 如果大于阈值，那么该点为外点
	// 如果是内点，那么根据论文(2)：ρ（d^2） = Γ-d^2,将分数累加
        if(chiSquare1>th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1
        // 将图像1中的点通过单应矩阵投影到图像2中，并计算对应的像素位置误差平方和
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

	// 如果像素位置平方和大于阈值，那么就当做外点剔除
	// 如果是内点，那么根据论文(2)：ρ（d^2） = Γ-d^2,将分数累加
        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

// 与单应性矩阵的分数计算相似，但是反投影到另一帧图像的对应点时有区别，用了极线，详见多视图几何中文版第159页对极几何内容
float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    // 基于卡方检验计算出的距离阈值（假设测量有一个像素的偏差）
    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // 我们先求第一帧图像的关键点(u1,v1)对应在第二帧图像中的极线
	// l2=F21x(u1,v1,1)^T=(a2,b2,c2)
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

	// 第二帧图像中的关键点到极线的距离：d=fabs(Ax+By+C)/sqrt(A^2+B^2)
	// 这里是平方
        const float num2 = a2*u2+b2*v2+c2;
        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =F21^T*x2=(a1,b1,c1)
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;
        // 点到线的几何距离的平方
        const float squareDist2 = num1*num1/(a1*a1+b1*b1);
 
        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

// 基本矩阵中恢复得到R和t
// 由于相机内参已知，所以可以结合内参转换乘本质矩阵E，再分解本质矩阵
// 关于基础矩阵和本质矩阵之间的关系以及分解本质矩阵求解位姿的方法，详见多视图几何中文版第173页
bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    // E = K^T*F*K
    cv::Mat E21 = K.t()*F21*K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    // 分解本质矩阵E，会得到四个结果
    DecomposeE(E21,R1,R2,t);  

    // 得到的两个不同的平移向量
    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    // 对四个解进行验证得到最合理的R和t
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    // 方法与单应矩阵比较分解的结果相同
    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    // 得到结果匹配最多的对应解
    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    // minTriangulated为可以三角化恢复三维点的个数，设置为50个
    // 在匹配的内点数量的90%与最小阈值之间选择大的那个，目的是保证可以三角化的匹配点数量
    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    // 如果未达到指定的最小可以三角化的匹配点数，返回失败
    // 四个结果中如果没有明显的最优结果，则返回失败
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    // 四个结果中挑选一个最合理的结果
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

// 从单应性矩阵中恢复R和T，用的方法与SVO相同
// 详见论文： Motion and structure from motion in a piecewise planar environment
bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;
    
    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    cv::Mat invK = K.inv();
    // K*A*K^-1 = H
    cv::Mat A = invK*H21*K;

    cv::Mat U,w,Vt,V;
    // 对矩阵A进行SVD分解
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();
    
    // s=|U|*|V|
    float s = cv::determinant(U)*cv::determinant(Vt);
    
    // d1,d2和d3分别是A*A^t的特征值的开方
    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    // 如果d1与d2 或者d2与d3 数值可以近似相等，那么就认为是非正常情况，返回错误标志
    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    // 会有八种结果
    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    // 其中x2=0
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    // 计算得到sin（theta）
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);
    // 计算得到cos（theta）
    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    // 计算旋转矩阵和平移向量
    // 由于ε1、ε2=1或者-1，所以分为四种情况（主要是旋转矩阵）
    //      | ctheta      0   -aux_stheta|       | aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      | aux_stheta  0    ctheta    |       |-aux3|

    //      | ctheta      0    aux_stheta|       | aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      |-aux_stheta  0    ctheta    |       | aux3|

    //      | ctheta      0    aux_stheta|       |-aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      |-aux_stheta  0    ctheta    |       |-aux3|

    //      | ctheta      0   -aux_stheta|       |-aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      | aux_stheta  0    ctheta    |       | aux3|
    for(int i=0; i<4; i++)
    {
        // 得到旋转矩阵R
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

	// 上面计算得到的Rp为R'，而R'=s*U^T*R*V,所以R = s*U*Rp*V^T
        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

	// 得到平移向量t
        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

	// 上面计算得到的tp为t'，而t'=U^T*t,所以t = U*tp
        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

	// 得到n',所以n=V*n'
        cv::Mat n = V*np;
	// 如果计算小于0，则取反
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    // 与d'=d2相似，只不过代入公式不同
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }


    // 从上面计算的8组结果中选择最合理的解
    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for(size_t i=0; i<8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;
	// 得到符合要求的点对数量，三维点信息
	// 通过三角化的点的视差角以及重投影误差来判断当前的分解R t是否合理
        int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

	// 保留最优的和次优的结果
	// 其中parallaxi为角度制下的视差角， vbTriangulatedi表示当前匹配点对的视差角满足要求的标志位
        if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }

    // 如果满足次有的关键点数小于0.75倍的最优解，并且minParallax = 1， minTriangulated = 50阈值条件满足且90%以上的点
    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        // 将R和t分别赋给R21和t21，并将三维点坐标以及视角符合要求的关键点对保存到相应的变量中
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }
    // 如果不符合相关条件，就表示解算的值有问题
    return false;
}

// 作用： 通过给定的投影矩阵P1,P2和图像上的点kp1,kp2，从而恢复三维空间中的3D点的坐标
// 采用了简单的线性三角形法恢复三维点坐标，原理详见多视图几何中文版第217页
// x' = P'X  x = PX（其中x和x'为像素坐标系上的坐标，X为要求的三维世界坐标系坐标）
// 它们都属于 x = aPX模型
//                         |X|
// |x|     |p1 p2  p3  p4 ||Y|     |x|    |--p0--||.|
// |y| = a |p5 p6  p7  p8 ||Z| ===>|y| = a|--p1--||X|
// |z|     |p9 p10 p11 p12||1|     |z|    |--p2--||.|
// 采用DLT的方法：x叉乘PX = 0
// |yp2 -  p1|     |0|
// |p0 -  xp2| X = |0|
// |xp1 - yp0|     |0|
// 两个点:
// |yp2   -  p1  |     |0|
// |p0    -  xp2 | X = |0| ===> AX = 0
// |y'p2' -  p1' |     |0|
// |p0'   - x'p2'|     |0|
// 变成程序中的形式：
// |xp2  - p0 |     |0|
// |yp2  - p1 | X = |0| ===> AX = 0
// |x'p2'- p0'|     |0|
// |y'p2'- p1'|     |0|
void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    // 计算得到A的第四行（最后一行），并转置转换为列向量
    x3D = vt.row(3).t();
    // 取前三行的数据作为三维点坐标，除以最后一行的全局缩放因子
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
    
    // 这里三角化过程中恢复的3D点深度取决于 t 的尺度，但该尺度并不是整个单目SLAM的尺度
    // 后面的相关函数会对该深度进行缩放，反过来还会影响t的值
}

// 函数输入：vKeys（特征点在图像上的坐标）、vNormalizedPoints（特征点归一化后的坐标）和T（将特征点归一化变换矩阵）
// 平面归一化变换，详见多视图几何中文版第67页
void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    // meanX和meanY分别为所有关键点x和y的平均值
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    // 平移变换，vKeys点减去中心坐标，使x坐标和y坐标均值分别为0（质心位于原点）
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

	// meanDevX和meanDevY为所有平移变换后关键点x和y的总和
	meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    // 平均x和y
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    // 尺度缩放，使得平均距离为根号2
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }
    
    // T为归一化变换矩阵，由平移和缩放组成，详见多视图几何中文版第193页
    // |sX  0  -meanx*sX|
    // |0   sY -meany*sY|
    // |0   0      1    |
    // vNormalizedPoints = T*vKeys
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

// 采用了cheirality check，从而进一步找出F分解后最合适的解
int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(vKeys1.size(),false);
    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    // 得到一个相机的投影矩阵，并以第一个相机的光心作为世界坐标系
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    // 得到第二个相机的投影矩阵
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    // 第二个相机的光心在世界坐标系下的坐标
    cv::Mat O2 = -R.t()*t;

    int nGood=0;

    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        // 不是匹配的点对，则直接跳过
        if(!vbMatchesInliers[i])
            continue;

	// 取匹配点对，分别保存至kp1和kp2
        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;

	// 三角化匹配点对，恢复对应的三维点p3dC1
        Triangulate(kp1,kp2,P1,P2,p3dC1);

	// 如果三维点p3dC1的任何一维坐标无穷大，那么该点为坏点，跳过且置相关标志位为false
	// isfinite： 检测值是否有限（Finite）
        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        // 计算视差角的余弦值，即点X与摄像机1的光心射线和点X与摄像机2的光心的射线构成的角
        cv::Mat normal1 = p3dC1 - O1;
	// 向量模长
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);
        // cos(theta) = a.b/(norm(a)*norm(b))
        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
	// 如果该点在第一个相机坐标系下的深度是负数，即跑到摄像机后面， 视差角接近于0 就跳过
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
	// 第二个相机坐标系下的深度为负数，不符合
        cv::Mat p3dC2 = R*p3dC1+t;
   
        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
	// 计算3D点在第一帧图像上的投影误差
	// 
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);
        // th2 = 4.0*mSigma2
        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
	// 计算3D点在第二帧图像上的投影误差
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        if(squareError2>th2)
            continue;
        // 保存视差角余弦
        vCosParallax.push_back(cosParallax);
	// 将三维点坐标也保存
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
	// 好的结果计数器加一
        nGood++;
        // 角度不接近于1时，对应的匹配点保存
        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    // 如果该组R t下有好的结果点，那么就对视差角余弦值排序（从小到大，对应角度为从大到小）
    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());
        // trick： 排序后并没有取最小的视差角，取一个较小的视差角
        size_t idx = min(50,int(vCosParallax.size()-1));
	// 转换为角度制
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}

// 通过SVD分解本质矩阵，得到四组解：[R1,t],[R1,-t],[R2,t],[R2,-t]
void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    // 对本质矩阵进行SVD分解，
    // E：3*3  u:3*3 w:3*3 vt:3*3
    cv::SVD::compute(E,w,u,vt);
    // 取u的最后一列
    u.col(2).copyTo(t);
    t=t/cv::norm(t);  // 对 t 有归一化，但是这个地方并没有决定单目整个SLAM过程的尺度

    //       |0 -1  0|
    //   W = |1  0  0|
    //       |0  0  1|
    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    // R1=U*W*v^T
    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;
    
    // R2=U*W^T*v^T
    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

} //namespace ORB_SLAM
