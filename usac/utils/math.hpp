#ifndef USAC_UTILS_MATH_H
#define USAC_UTILS_MATH_H

#include "../precomp.hpp"

class Time {
public:	
    long minutes;
    long seconds;
    long milliseconds;
    long microseconds;

    friend std::ostream& operator<< (std::ostream& stream, const Time * time) {
        return stream << time->seconds << " secs " << time->milliseconds << " ms " <<
               time->microseconds << " mcs\n";
    }
};

class Math {
public:
    static bool inverse3x3 (cv::Mat& A);
    static bool inverse3x3 (const cv::Mat& A, cv::Mat& A_inv);

    static float fast_pow (float n, int k);

    static int fast_factorial (int n);

    static void splitTime (Time * time, long time_mcs);
    static bool haveCollinearPoints (const float * const points, const int * const sample, unsigned int sample_size);
    static bool isPointsClosed (const float * const points, const int * const sample, unsigned int sample_size, float min_dist=5);

    static cv::Mat getSkewSymmetric(const cv::Mat &v) {
        return (cv::Mat_<float>(3,3) << 0, -v.at<float>(2), v.at<float>(1),
                                        v.at<float>(2), 0, -v.at<float>(0),
                                       -v.at<float>(1), v.at<float>(0), 0);
    }
    static cv::Mat getCrossProductDim3(const cv::Mat &a, const cv::Mat &b) {
        return (cv::Mat_<float>(3,1) << a.at<float>(1) * b.at<float>(2) - a.at<float>(2) * b.at<float>(1),
                                        a.at<float>(2) * b.at<float>(0) - a.at<float>(0) * b.at<float>(2),
                                        a.at<float>(0) * b.at<float>(1) - a.at<float>(1) * b.at<float>(0));
    }
    static unsigned int getRank (const cv::Mat &A, float threshold) {
        cv::Mat w; // vector of singular values.
        cv::SVD::compute(A, w);
        unsigned int max_rank = std::min(A.rows, A.cols);
        unsigned int rank = max_rank;
        for (unsigned int i = 0; i < max_rank; i++) {
            if (w.at<float>(i) < threshold) {
                rank--;
            }
        }
        return rank;
    }

};

#endif //USAC_UTILS_MATH_H
