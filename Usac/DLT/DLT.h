#ifndef RANSAC_DLT_H
#define RANSAC_DLT_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

// Direct Linear Transformation
//void DLT (cv::InputArray pts1, cv::InputArray pts2, cv::Mat &H) {
void DLT (cv::InputArray pts, const int * const sample, int sample_number, cv::Mat &H) {

    /*
     * mat array N x 2
     * x1 y1
     * x2 y2
     * ...
     * xN yN
     *
     * float array 2N x 1
     * x1
     * y1
     * x2
     * y2
     * ...
     * xN
     * yN
     */

//    float * points1 = (float *) pts1.getMat().data;
//    float * points2 = (float *) pts2.getMat().data;

    float * const points = (float *) pts.getMat().data;

    float *points1 = new float[2*sample_number], *points2 = new float[2*sample_number];

    for (int i = 0; i < sample_number; i++) {
        points1[2*i] = points[4*sample[i]];
        points1[2*i+1] = points[4*sample[i]+1];
        points2[2*i] = points[4*sample[i]+2];
        points2[2*i+1] = points[4*sample[i]+3];
    }
    
    int NUMP = sample_number; //pts1.getMat().rows;

    float x1, y1, x2, y2;

//    std::cout << "NUMP = "<< points1.size << "\n";

    cv::Mat_<float> A (2*NUMP, 9), w, u, vt;
    float * A_ptr = (float *) A.data;

    for (int i = 1; i <= NUMP; i++) {
        x1 = points1[2*(i-1)];
        y1 = points1[2*i-1];

        x2 = points2[2*(i-1)];
        y2 = points2[2*i-1];

        (*A_ptr++) = -x1;
        (*A_ptr++) = -y1;
        (*A_ptr++) = -1;
        (*A_ptr++) = 0;
        (*A_ptr++) = 0;
        (*A_ptr++) = 0;
        (*A_ptr++) = x2*x1;
        (*A_ptr++) = x2*y1;
        (*A_ptr++) = x2;

        (*A_ptr++) = 0;
        (*A_ptr++) = 0;
        (*A_ptr++) = 0;
        (*A_ptr++) = -x1;
        (*A_ptr++) = -y1;
        (*A_ptr++) = -1;
        (*A_ptr++) = y2*x1;
        (*A_ptr++) = y2*y1;
        (*A_ptr++) = y2;
    }

    /*
     * src	decomposed matrix
     * w 	calculated singular values
     * u	calculated left singular vectors
     * vt	transposed matrix of right singular values
     */
    cv::SVD::compute(A, w, u, vt);

    H = cv::Mat_<float>(vt.row(vt.rows-1).reshape (3,3));
}



#endif // RANSAC_DLT_H