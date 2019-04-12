// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

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
};

#endif //USAC_UTILS_MATH_H
