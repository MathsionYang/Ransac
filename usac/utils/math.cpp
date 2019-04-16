// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "math.hpp"

/*
 * Fast finding inverse matrix 3x3
 * A^-1 = adj(A)/det(A)
 */
bool Math::inverse3x3 (cv::Mat& A) {
//    assert (A.rows == 3 && A.cols == 3);

    float * A_ptr = (float *) A.data;
    float a11 = A_ptr[0];
    float a12 = A_ptr[1];
    float a13 = A_ptr[2];
    float a21 = A_ptr[3];
    float a22 = A_ptr[4];
    float a23 = A_ptr[5];
    float a31 = A_ptr[6];
    float a32 = A_ptr[7];
    float a33 = A_ptr[8];

    float detA = a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31;

    if (detA == 0) {
        std::cout << "\033[1;31mDeterminant of A is 0\033[0m\n";
        return false;
    }
//    else if (fabsf(detA) < 0.0000001) {
//        std::cout << detA << "\n";
//        std::cout << "\033[1;33mDeterminant of A is very small\033[0m\n";
//    }

    A_ptr[0] = (a22*a33 - a23*a32)/detA;
    A_ptr[1] = (a13*a32 - a12*a33)/detA;
    A_ptr[2] = (a12*a23 - a13*a22)/detA;
    A_ptr[3] = (a23*a31 - a21*a33)/detA;
    A_ptr[4] = (a11*a33 - a13*a31)/detA;
    A_ptr[5] = (a13*a21 - a11*a23)/detA;
    A_ptr[6] = (a21*a32 - a22*a31)/detA;
    A_ptr[7] = (a12*a31 - a11*a32)/detA;
    A_ptr[8] = (a11*a22 - a12*a21)/detA;

    return true;
}

bool Math::inverse3x3 (const cv::Mat& A, cv::Mat& A_inv){
    A_inv = A.clone();
    inverse3x3(A_inv);
}


/*
 *
 * Declare fast_pow function because c++ pow is very slow
 * https://stackoverflow.com/questions/41072787/why-is-powint-int-so-slow/41072811
 * pow(x,y) = e^(y log(x))
 *
 */
float Math::fast_pow (float n, int k) {
    float res = n;
    while (k > 1) {
        res *= n; k--;
    }
    return res;
}

int Math::fast_factorial (int n) {
    int res = n;
    while (n > 2) {
        res *= --n;
    }
    return res;
}


/*
 * @points Nx4 array: x1 y1 x2 y2
 * @sample Mx1 array
 */
bool Math::haveCollinearPoints (const float * const points, const int * const sample, unsigned int sample_size) {
    unsigned int last_pt_idx = 4*sample[sample_size-1];
    float last_pt_x= points[last_pt_idx  ];
    float last_pt_y = points[last_pt_idx+1];
    float last_pt_X = points[last_pt_idx+2];
    float last_pt_Y = points[last_pt_idx+3];

    // Checks if no more than 2 points are on the same line
    //     |x1 y1 1|
    // det |x2 y2 1| = 0
    //     |x3 y3 1|
    float x1, y1, x2, y2, X1, Y1, X2, Y2;
    unsigned pt_idx;
    for (unsigned int j = 0; j < sample_size-1; j++) {
        pt_idx = 4*sample[j];
        x1 = points[pt_idx  ];
        y1 = points[pt_idx+1];
        X1 = points[pt_idx+2];
        Y1 = points[pt_idx+3];

        for (unsigned int k = 0; k < j; k++){
            pt_idx = 4*sample[k];
            x2 = points[pt_idx  ];
            y2 = points[pt_idx+1];
            X2 = points[pt_idx+2];
            Y2 = points[pt_idx+3];

            if (fabsf(x1*(y2-last_pt_y) + x2*(last_pt_y-y1) + last_pt_x*(y1-y2)) < FLT_EPSILON ||
                fabsf(X1*(Y2-last_pt_Y) + X2*(last_pt_Y-Y1) + last_pt_X*(Y1-Y2)) < FLT_EPSILON) return true;
        }
    }
    return false;
}

/*
 * @points Nx4 array: x1 y1 x2 y2
 * @sample Mx1 array
 */
bool Math::isPointsClosed (const float * const points, const int * const sample, unsigned int sample_size, float min_dist) {
    float x1, y1, X1, Y1, x2, y2, X2, Y2;
    unsigned pt_idx;
    for (unsigned int i = 0; i < sample_size; i++) {
        pt_idx = 4*sample[i];
        x1 = points[pt_idx  ];
        y1 = points[pt_idx+1];
        X1 = points[pt_idx+2];
        Y1 = points[pt_idx+3];
        for (unsigned int j = i+1; j < sample_size; j++) {
            pt_idx = 4 * sample[j];
            x2 = points[pt_idx    ];
            y2 = points[pt_idx + 1];
            X2 = points[pt_idx + 2];
            Y2 = points[pt_idx + 3];
            if (sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) < min_dist ||
                sqrt((X1 - X2) * (X1 - X2) + (Y1 - Y2) * (Y1 - Y2)) < min_dist)
                return true;
        }
    }
    return false;
}

void Math::splitTime (Time * time, long time_mcs) {
    time->microseconds = time_mcs % 1000;
    time->milliseconds = ((time_mcs - time->microseconds)/1000) % 1000;
    time->seconds = ((time_mcs - 1000*time->milliseconds - time->microseconds)/(1000*1000)) % 60;
    time->minutes = ((time_mcs - 60*1000*time->seconds - 1000*time->milliseconds - time->microseconds)/(60*1000*1000)) % 60;
}