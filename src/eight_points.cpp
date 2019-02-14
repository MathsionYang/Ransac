// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "../include/opencv2/usac/fundamental_solver.hpp"
#include "../include/opencv2/usac/dlt.hpp"

bool cv::usac::FundamentalSolver::EightPointsAlgorithm (const int * const sample, unsigned int sample_number, cv::Mat &F) {

    cv::Mat_<float> T1, T2, norm_points;
    cv::usac::GetNormalizingTransformation(points, norm_points, sample, sample_number, T1, T2);

    const float * const norm_points_ptr = (float *) norm_points.data;

    // form a linear system Ax=0: for each selected pair of points m1 & m2,
    // the row of A(=a) represents the coefficients of equation: (m2, 1)'*F*(m1, 1) = 0
    // to save computation time, we compute (At*A) instead of A and then solve (At*A)x=0.
    float x1, x2, y1, y2;
    unsigned int norm_points_idx;

    // -------------- 8 points with SVD ---------------------
    cv::Mat_<float> A(sample_number, 9);
    auto * A_ptr = (float *) A.data;
    for (unsigned int i = 0; i < sample_number; i++) {
        norm_points_idx = 4*i;
        x1 = norm_points_ptr[norm_points_idx];
        y1 = norm_points_ptr[norm_points_idx+1];
        x2 = norm_points_ptr[norm_points_idx+2];
        y2 = norm_points_ptr[norm_points_idx+3];
        (*A_ptr++) = x2*x1;
        (*A_ptr++) = x2*y1;
        (*A_ptr++) = x2;
        (*A_ptr++) = y2*x1;
        (*A_ptr++) = y2*y1;
        (*A_ptr++) = y2;
        (*A_ptr++) = x1;
        (*A_ptr++) = y1;
        (*A_ptr++) = 1;
    }

    cv::Mat_<float> U, S, Vt;
    cv::SVD::compute(A, S, U, Vt);
    // -----------------------------------------------------

    if (Vt.empty())
        return false;

    F = cv::Mat_<float> (Vt.row(Vt.rows-1).reshape (3,3));

    auto * t2 = (float *) T2.data;

    // Transpose T2
    t2[6] = t2[2];
    t2[7] = t2[5];
    t2[2] = 0;
    t2[5] = 0;

    F = T2 * F * T1;

    // normalize by f33
    if (fabs(F.at<float>(2,2)) > FLT_EPSILON) {
        F = F / F.at<float>(2, 2);
    }

    return true;
}

bool cv::usac::FundamentalSolver::EightPointsAlgorithmEigen (const int * const sample, unsigned int sample_number, cv::Mat &F) {

    cv::Mat_<float> T1, T2, norm_points;
    cv::usac::GetNormalizingTransformation(points, norm_points, sample, sample_number, T1, T2);

    const float * const norm_points_ptr = (float *) norm_points.data;

    float x1, x2, y1, y2;
    unsigned int norm_points_idx;

    // ------- 8 points algorithm with Eigen and covariance matrix --------------
    float a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 1};
    cv::Mat_ <float> covA(9, 9, float (0));
    float * covA_ptr = (float *) covA.data;
    for (unsigned int i = 0; i < sample_number; i++) {
        norm_points_idx = 4*i;
        x1 = norm_points_ptr[norm_points_idx];
        y1 = norm_points_ptr[norm_points_idx+1];
        x2 = norm_points_ptr[norm_points_idx+2];
        y2 = norm_points_ptr[norm_points_idx+3];
        a[0] = x2*x1;
        a[1] = x2*y1;
        a[2] = x2;
        a[3] = y2*x1;
        a[4] = y2*y1;
        a[5] = y2;
        a[6] = x1;
        a[7] = y1;

        // calculate covariance for eigen
        for (unsigned int row = 0; row < 9; row++) {
            for (unsigned int col = row; col < 9; col++) {
                covA_ptr[row*9+col] += a[row]*a[col];
            }
        }
    }

    /*
     * Copy upper triangle of A to lower triangle of A.
     * symmetric covariance matrix
     */
    for (unsigned int row = 1; row < 9; row++) {
        for (unsigned int col = 0; col < row; col++) {
            covA_ptr[row*9+col] = covA_ptr[col*9+row];
        }
    }

    cv::Mat_<float> D, Vt;
    cv::eigen(covA, D, Vt);
// -------------------------------------------------------------

    if (Vt.empty()) {
        return false;
    }

    F = cv::Mat_<float> (Vt.row(Vt.rows-1 /* 8 */).reshape (3,3));

    auto * t2 = (float *) T2.data;

    t2[6] = t2[2];
    t2[7] = t2[5];
    t2[2] = 0;
    t2[5] = 0;

    F = T2 * F * T1;

    if (fabs(F.at<float>(2,2)) > FLT_EPSILON) {
        F = F / F.at<float>(2, 2);
    }

    return true;
}