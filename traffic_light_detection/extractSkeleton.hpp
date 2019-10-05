/**
* @author ZiYao Xiao
* @email aixiaoyaowudi@gmail.com
* @reference https://blog.csdn.net/xukaiwen_2016/article/details/53135866
* @reference https://translate.google.com
*/
#ifndef _GLIBCXX_EXTRACT_SKELETON_HPP_
#define _GLIBCXX_EXTRACT_SKELETON_HPP_ 1

#pragma GCC system_header
#pragma once

#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <iostream>  
#include <vector>  
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace std;

namespace sidewalkTrafficLight _GLIBCXX_VISIBILITY(default)
{
    class extractSkeleton{
    public:
        extractSkeleton();
        //~extractSkeleton() = default;
        /**
        * @brief Refine the input image, bones
        * @param src is the input image,8-bit grayscale image format processed with the cvThreshold function，Only 0 and 1 in the element, 1 means there is an element, 0 means blank.Length and width must be less than 1000
        * @param maxIterations limits the number of iterations. If not, the default is -1, which means that the number of iterations is not limited until the final result is obtained.
        * @return For the output image after src refinement, the format is the same as the src format. Only 0 and 1 are in the element, 1 means that there is an element, and 0 means blank.
        */
        cv::Mat thinImage(const cv::Mat & src, const int);
        /**
        * @brief filters the skeletal map data to achieve at least one blank pixel between two points
        * @param thinSrc is the input bone image, 8-bit gray image format, only 0 and 1 in the element, 1 means there is an element, 0 means blank
        */
        void filterOver(cv::Mat thinSrc);
        /**
        * @brief Find endpoints and intersections from filtered bone images
        * @param thinSrc is the input filtered skeleton image, 8-bit grayscale image format, only 0 and 1 in the element, 1 means there is an element, 0 means blank
        * @param raudis convolution radius, with the current pixel point center, determine whether the point is an endpoint or intersection in the circle
        * @param thresholdMax crosspoint threshold, greater than this value is the intersection
        * @param thresholdMin endpoint threshold, less than this value is the endpoint
        * @return is the output image after src refinement, the format is the same as src format, only 0 and 1 in the element, 1 means there is an element, 0 means blank
        */
        std::vector<cv::Point> getPoints(const cv::Mat&, unsigned int, unsigned int, unsigned int );
        /**
        * @brief extracts key points in endpoints and intersections
        * @param src is a refined and filtered skeletonized image
        * @param points are the original endpoints and intersections
        * @return Simplified endpoints and intersections
        */
        std::vector<cv::Point> clearPoints(const cv::Mat, std::vector<cv::Point>);
        /**
        * @brief calculate the best minPointDist for the image
        * @param src is the original source image
        * @return void
        */
        void calculateMinPointDist(const cv::Mat);
    private:
        /**
        * @brief finds the distance between the source and other points
        * @param src is a refined and filtered skeletonized image
        * @param points is the source point
        * @return distance between source point and other points
        */
        std::vector<std::pair<cv::Point, long long>> connectPoint(const cv::Mat, const cv::Point);
        /**
        * @brief refer to the position of points
        */
        long long* findPoint = new long long[1000001];
        /**
        * @brief refer to whether the point has been visited
        */
        bool* visitPoint = new bool[1000001];
        /**
        * @brief the length of findPoint
        */
        long long findPointLength = 1000001;
        /**
        * @brief the length of visitPoint
        */
        long long visitPointLength = 1000001;
    protected:
        /**
        * @brief refer to the min distance of different points
        */
        long long minPointDist=0;
    };
    cv::Mat extractSkeleton::thinImage(const cv::Mat & src, const int maxIterations = -1)
    {
        assert(src.type() == CV_8UC1);
        cv::Mat dst;
        int width = src.cols;
        int height = src.rows;
        src.copyTo(dst);
        int count = 0; 
        while (true)
        {
            count++;
            if (maxIterations != -1 && count > maxIterations)
                break;
            std::vector<uchar *> mFlag;   
            for (int i = 0; i < height; ++i)
            {
                uchar * p = dst.ptr<uchar>(i);
                for (int j = 0; j < width; ++j)
                {
                    uchar p1 = p[j];
                    if (p1 != 1) continue;
                    uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
                    uchar p8 = (j == 0) ? 0 : *(p + j - 1);
                    uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
                    uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
                    uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
                    uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
                    uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
                    uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
                    if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
                    {
                        int ap = 0;
                        if (p2 == 0 && p3 == 1) ++ap;
                        if (p3 == 0 && p4 == 1) ++ap;
                        if (p4 == 0 && p5 == 1) ++ap;
                        if (p5 == 0 && p6 == 1) ++ap;
                        if (p6 == 0 && p7 == 1) ++ap;
                        if (p7 == 0 && p8 == 1) ++ap;
                        if (p8 == 0 && p9 == 1) ++ap;
                        if (p9 == 0 && p2 == 1) ++ap;
     
                        if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
                        {  
                            mFlag.push_back(p + j);
                        }
                    }
                }
            }
            for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
            {
                **i = 0;
            }
            if (mFlag.empty())
            {
                break;
            }
            else
            {
                mFlag.clear(); 
            }
            for (int i = 0; i < height; ++i)
            {
                uchar * p = dst.ptr<uchar>(i);
                for (int j = 0; j < width; ++j)
                {
                    uchar p1 = p[j];
                    if (p1 != 1) continue;
                    uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
                    uchar p8 = (j == 0) ? 0 : *(p + j - 1);
                    uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
                    uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
                    uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
                    uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
                    uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
                    uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
     
                    if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
                    {
                        int ap = 0;
                        if (p2 == 0 && p3 == 1) ++ap;
                        if (p3 == 0 && p4 == 1) ++ap;
                        if (p4 == 0 && p5 == 1) ++ap;
                        if (p5 == 0 && p6 == 1) ++ap;
                        if (p6 == 0 && p7 == 1) ++ap;
                        if (p7 == 0 && p8 == 1) ++ap;
                        if (p8 == 0 && p9 == 1) ++ap;
                        if (p9 == 0 && p2 == 1) ++ap;
     
                        if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
                        {
                            mFlag.push_back(p + j);
                        }
                    }
                }
            } 
            for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
            {
                **i = 0;
            }  
            if (mFlag.empty())
            {
                break;
            }
            else
            {
                mFlag.clear();
            }
        }
        return dst;
    }

    void extractSkeleton::filterOver(cv::Mat thinSrc)
    {
        assert(thinSrc.type() == CV_8UC1);
        int width = thinSrc.cols;
        int height = thinSrc.rows;
        for (int i = 0; i < height; ++i)
        {
            uchar * p = thinSrc.ptr<uchar>(i);
            for (int j = 0; j < width; ++j)
            {  
                uchar p1 = p[j];
                if (p1 != 1) continue;
                uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
                uchar p8 = (j == 0) ? 0 : *(p + j - 1);
                uchar p2 = (i == 0) ? 0 : *(p - thinSrc.step + j);
                uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - thinSrc.step + j + 1);
                uchar p9 = (i == 0 || j == 0) ? 0 : *(p - thinSrc.step + j - 1);
                uchar p6 = (i == height - 1) ? 0 : *(p + thinSrc.step + j);
                uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + thinSrc.step + j + 1);
                uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + thinSrc.step + j - 1);
                if (p2 + p3 + p8 + p9 >= 1)
                {
                    p[j] = 0;
                }
            }
        }
    }

    std::vector<cv::Point> extractSkeleton::getPoints(const cv::Mat &thinSrc, unsigned int raudis = 4, unsigned int thresholdMax = 6, unsigned int thresholdMin = 4)
    {
        assert(thinSrc.type() == CV_8UC1);
        int width = thinSrc.cols;
        int height = thinSrc.rows;
        cv::Mat tmp;
        thinSrc.copyTo(tmp);
        std::vector<cv::Point> points;
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                if (*(tmp.data + tmp.step * i + j) == 0)
                {
                    continue;
                }
                int count=0;
                for (int k = i - raudis; k < i + raudis+1; k++)
                {
                    for (int l = j - raudis; l < j + raudis+1; l++)
                    {
                        if (k < 0 || l < 0||k>height-1||l>width-1)
                        {
                            continue;
                            
                        }
                        else if (*(tmp.data + tmp.step * k + l) == 1)
                        {
                            count++;
                        }
                    }
                }
     
                if (count > thresholdMax||count<thresholdMin)
                {
                    cv::Point point(j, i);
                    points.push_back(point);
                }
            }
        }
        return points;
    }

    std::vector<cv::Point> extractSkeleton::clearPoints(const cv::Mat src, std::vector<cv::Point> points)
    {
        std::vector<cv::Point>::iterator it = points.begin();
        for (long long pos = 1; it!=points.end(); it++, pos++)
        {
            findPoint[1000 * it->x + it->y ] = pos;
        }
        it = points.begin();
        for (;it!=points.end() ; it++)
        {
            if(findPoint[it->x * 1000 + it->y]==0) continue;
            std::vector<std::pair<cv::Point, long long>> waitConnect = connectPoint(src, *it);
            std::vector<std::pair<cv::Point, long long>>::iterator _it = waitConnect.begin();
            for (; _it!=waitConnect.end(); _it++)
            {
                if ( _it->second <= minPointDist)
                {
                    findPoint[it->x * 1000 + it->y] = 0;
                    break;
                }
            }
        }
        std::vector<cv::Point> ans;
        it = points.begin();
        for (long long pos=0; it!=points.end(); it++, pos++)
        {
            if ( findPoint[it->x * 1000 + it->y])
            {
                ans.push_back(*it);
            }
        }
        it = points.begin();
        for (long long pos = 0; it!=points.end(); it++, pos++)
        {
            findPoint[1000 * it->x + it->y ] = 0;
        }
        return ans;
    }

    std::vector<std::pair<cv::Point, long long>> extractSkeleton::connectPoint(const cv::Mat src, const cv::Point point)
    {
        long long direction[49][2];
        for(long long i = 0; i < 7; i++){for(long long j = 0; j < 7; j++){direction[i*7+j][0] = i+1LL-4LL;direction[i*7+j][1] = j+1LL-4LL;}}
        visitPoint[point.x * 1000 + point.y] = true ;
        std::vector<std::pair<cv::Point, long long>> waitProcess, ans;
        waitProcess.push_back(std::pair<cv::Point, long long>(point, 0));
        while (waitProcess.size())
        {
            cv::Point tmp = waitProcess.front().first;
            long long dist = waitProcess.front().second;
            if ( findPoint[tmp.x * 1000 + tmp.y] && (tmp.x != point.x || tmp.y != point.y))
            {
                ans.push_back(std::pair<cv::Point, long long>(tmp, dist));
                waitProcess.erase(waitProcess.begin());
                continue;
            }
            waitProcess.erase(waitProcess.begin());
            for (long long _p=0;_p<49;_p++)
            {
                tmp.x += direction[_p][0];
                tmp.y += direction[_p][1];
                if (src.ptr<uchar>(tmp.y)[tmp.x] > 0 && !visitPoint[tmp.x * 1000 + tmp.y])
                {
                    visitPoint[tmp.x * 1000 + tmp.y] = true;
                    waitProcess.push_back(std::pair<cv::Point, long long>(tmp,dist+1));
                }
                tmp.x -= direction[_p][0];
                tmp.y -= direction[_p][1];
            }
        }
        memset(visitPoint, 0, sizeof(bool) * visitPointLength);
        return ans;
    }

    void extractSkeleton::calculateMinPointDist(const cv::Mat src)
    {
        minPointDist = std::max(src.cols, src.rows) / 40;
        //minPointDist = 0;
    }

    extractSkeleton::extractSkeleton()
    {
        memset(findPoint, 0, sizeof(long long) * findPointLength);
        memset(visitPoint, 0, sizeof(bool) * visitPointLength);
    }
    // extractSkeleton::~extractSkeleton()
    // {
    //     delete [] findPoint;
    //     delete [] visitPoint;
    // }
}

#endif