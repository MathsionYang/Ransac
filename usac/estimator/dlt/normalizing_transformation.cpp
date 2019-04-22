#include "dlt.hpp"

void GetNormalizingTransformation (const float * const pts, cv::Mat& norm_points,
                                   const int * const sample, unsigned int sample_number, cv::Mat &T1, cv::Mat &T2) {

    float mean_pts1_x = 0, mean_pts1_y = 0, mean_pts2_x = 0, mean_pts2_y = 0;

    unsigned int smpl;
    for (unsigned int i = 0; i < sample_number; i++) {
        smpl = 4 * sample[i];

        mean_pts1_x += pts[smpl];
        mean_pts1_y += pts[smpl + 1];
        mean_pts2_x += pts[smpl + 2];
        mean_pts2_y += pts[smpl + 3];
    }

    mean_pts1_x /= sample_number;
    mean_pts1_y /= sample_number;
    mean_pts2_x /= sample_number;
    mean_pts2_y /= sample_number;

    float avg_dist1 = 0, avg_dist2 = 0, x1_m, y1_m, x2_m, y2_m;
    for (unsigned int i = 0; i < sample_number; i++) {
        smpl = 4 * sample[i];
        /*
         * Compute a similarity transform T that takes points xi
         * to a new set of points x̃i such that the centroid of
         * the points x̃i is the coordinate origin and their
         * average distance from the origin is √2
         *
         * origin O(0,0)
         * sqrt(x̃*x̃ + ỹ*ỹ) = sqrt(2)
         * ax*ax + by*by = 2
         */
        x1_m = pts[smpl    ] - mean_pts1_x;
        y1_m = pts[smpl + 1] - mean_pts1_y;
        x2_m = pts[smpl + 2] - mean_pts2_x;
        y2_m = pts[smpl + 3] - mean_pts2_y;

        avg_dist1 += sqrt (x1_m * x1_m + y1_m * y1_m);
        avg_dist2 += sqrt (x2_m * x2_m + y2_m * y2_m);
    }

    // scale
    avg_dist1 = M_SQRT2 / (avg_dist1 / sample_number);
    avg_dist2 = M_SQRT2 / (avg_dist2 / sample_number);

    /*
     * pts1_T1 = [ 1 0 -mean_pts1_x;
     *             0 1 -mean_pts1_y
     *             0 0 1]
     *
     * pts1_T2 = [ avg_dist1  0         0
     *             0         avg_dist1  0
     *             0          0         1]
     *
     * T1 = [avg_dist1  0          -mean_pts1_x*avg_dist1
     *       0         avg_dist1   -mean_pts1_y*avg_dist1
     *       0         0          1]
     *
     */

    T1 = (cv::Mat_<float>(3, 3) << avg_dist1, 0, -mean_pts1_x * avg_dist1,
                                    0, avg_dist1, -mean_pts1_y * avg_dist1,
                                    0, 0, 1);
    T2 = (cv::Mat_<float>(3, 3) << avg_dist2, 0, -mean_pts2_x * avg_dist2,
                                    0, avg_dist2, -mean_pts2_y * avg_dist2,
                                    0, 0, 1);

    auto *T1_ptr = (float *) T1.data;
    auto *T2_ptr = (float *) T2.data;

    norm_points = cv::Mat_<float>(sample_number, 4);

    auto *norm_points_ptr = (float *) norm_points.data;

    /*
     * Normalized points
     * Norm_img1_x1 Norm_img1_y1 Norm_img2_x1 Norm_img2_y1
     * Norm_img1_x2 Norm_img1_y2 Norm_img2_x2 Norm_img2_y2
     * ...
     * Norm_img1_xn Norm_img1_yn Norm_img2_xn Norm_img2_yn
     *
     * Npts1 = T1*pts1    3x3 * 3xN
     * Npts2 = T2*pts2    3x3 * 3xN
     *
     * Npts = [Npts1; Npts2]
     *
     * Fast T*pts multiplication below
     * We don't need third coordinate for points and third row for T,
     * because third column for output points is z(i) = 1
     *
     * N_x1 = T(1,1) * x1 + T(1,3)
     * N_y1 = T(2,2) * y1 + T(2,3)
     *
     * We don't need T(1,2) * y1 and T(2,2) * x1 because T(1,2) = T(2,1) = 0
     */
    unsigned int norm_pts_idx;
    for (unsigned int i = 0; i < sample_number; i++) {
        smpl = 4 * sample[i];
        norm_pts_idx = 4 * i;
        norm_points_ptr[norm_pts_idx    ] = T1_ptr[0] * pts[smpl    ] + T1_ptr[2]; // Norm_img1_xi
        norm_points_ptr[norm_pts_idx + 1] = T1_ptr[4] * pts[smpl + 1] + T1_ptr[5]; // Norm_img1_yi

        norm_points_ptr[norm_pts_idx + 2] = T2_ptr[0] * pts[smpl + 2] + T2_ptr[2]; // Norm_img2_xi
        norm_points_ptr[norm_pts_idx + 3] = T2_ptr[4] * pts[smpl + 3] + T2_ptr[5]; // Norm_img2_yi
    }
}



// Weighted Normalizing Transformation
void GetNormalizingTransformation (const float * const pts, cv::Mat& norm_points,
                                   const int * const sample, unsigned int sample_number, const float * const weights, cv::Mat &T1, cv::Mat &T2) {

    float mean_pts1_x = 0, mean_pts1_y = 0, mean_pts2_x = 0, mean_pts2_y = 0;
    unsigned int smpl;
    unsigned int wsmpl;
    for (unsigned int i = 0; i < sample_number; i++) {
        wsmpl = sample[i];
        smpl = 4 * wsmpl;

        mean_pts1_x += weights[wsmpl] * pts[smpl];
        mean_pts1_y += weights[wsmpl] * pts[smpl+1];
        mean_pts2_x += weights[wsmpl] * pts[smpl+2];
        mean_pts2_y += weights[wsmpl] * pts[smpl+3];
    }
    mean_pts1_x /= sample_number;
    mean_pts1_y /= sample_number;
    mean_pts2_x /= sample_number;
    mean_pts2_y /= sample_number;

    float avg_dist1 = 0, avg_dist2 = 0, x1_m, y1_m, x2_m, y2_m;
    for (unsigned int i = 0; i < sample_number; i++) {
        wsmpl = sample[i];
        smpl = 4 * wsmpl;

        x1_m = weights[wsmpl] * pts[smpl  ] - mean_pts1_x;
        y1_m = weights[wsmpl] * pts[smpl+1] - mean_pts1_y;
        x2_m = weights[wsmpl] * pts[smpl+2] - mean_pts2_x;
        y2_m = weights[wsmpl] * pts[smpl+3] - mean_pts2_y;

        avg_dist1 += sqrt(x1_m * x1_m + y1_m * y1_m);
        avg_dist2 += sqrt(x2_m * x2_m + y2_m * y2_m);
    }

    // scale
    avg_dist1 = M_SQRT2 / (avg_dist1 / sample_number);
    avg_dist2 = M_SQRT2 / (avg_dist2 / sample_number);

    T1 = (cv::Mat_<float>(3, 3) << avg_dist1, 0, -mean_pts1_x * avg_dist1,
            0, avg_dist1, -mean_pts1_y * avg_dist1,
            0, 0, 1);
    T2 = (cv::Mat_<float>(3, 3) << avg_dist2, 0, -mean_pts2_x * avg_dist2,
            0, avg_dist2, -mean_pts2_y * avg_dist2,
            0, 0, 1);

    auto *T1_ptr = (float *) T1.data;
    auto *T2_ptr = (float *) T2.data;

    norm_points = cv::Mat_<float>(sample_number, 4);

    auto *norm_points_ptr = (float *) norm_points.data;

    unsigned int norm_pts_idx;
    for (unsigned int i = 0; i < sample_number; i++) {
        smpl = 4 * sample[i];
        norm_pts_idx = 4 * i;
        norm_points_ptr[norm_pts_idx    ] = T1_ptr[0] * pts[smpl    ] + T1_ptr[2]; // Norm_img1_xi
        norm_points_ptr[norm_pts_idx + 1] = T1_ptr[4] * pts[smpl + 1] + T1_ptr[5]; // Norm_img1_yi

        norm_points_ptr[norm_pts_idx + 2] = T2_ptr[0] * pts[smpl + 2] + T2_ptr[2]; // Norm_img2_xi
        norm_points_ptr[norm_pts_idx + 3] = T2_ptr[4] * pts[smpl + 3] + T2_ptr[5]; // Norm_img2_yi
    }
}

// Weighted Normalizing Transformation
void GetNormalizingTransformation (const float * const pts, cv::Mat& norm_points,
                                   const int * const sample, unsigned int sample_number, const float * const weightsx, const float * const weightsy, cv::Mat &T1, cv::Mat &T2) {

    float mean_pts1_x = 0, mean_pts1_y = 0, mean_pts2_x = 0, mean_pts2_y = 0;
    unsigned int smpl;
    unsigned int wsmpl;
    for (unsigned int i = 0; i < sample_number; i++) {
        wsmpl = sample[i];
        smpl = 4 * wsmpl;

        mean_pts1_x += weightsx[wsmpl] * pts[smpl];
        mean_pts1_y += weightsy[wsmpl] * pts[smpl+1];
        mean_pts2_x += weightsx[wsmpl] * pts[smpl+2];
        mean_pts2_y += weightsy[wsmpl] * pts[smpl+3];
    }
    mean_pts1_x /= sample_number;
    mean_pts1_y /= sample_number;
    mean_pts2_x /= sample_number;
    mean_pts2_y /= sample_number;

    float avg_dist1 = 0, avg_dist2 = 0, x1_m, y1_m, x2_m, y2_m;
    for (unsigned int i = 0; i < sample_number; i++) {
        wsmpl = sample[i];
        smpl = 4 * wsmpl;

        x1_m = weightsx[wsmpl] * pts[smpl  ] - mean_pts1_x;
        y1_m = weightsy[wsmpl] * pts[smpl+1] - mean_pts1_y;
        x2_m = weightsx[wsmpl] * pts[smpl+2] - mean_pts2_x;
        y2_m = weightsy[wsmpl] * pts[smpl+3] - mean_pts2_y;

        avg_dist1 += sqrt(x1_m * x1_m + y1_m * y1_m);
        avg_dist2 += sqrt(x2_m * x2_m + y2_m * y2_m);
    }

    // scale
    avg_dist1 = M_SQRT2 / (avg_dist1 / sample_number);
    avg_dist2 = M_SQRT2 / (avg_dist2 / sample_number);

    T1 = (cv::Mat_<float>(3, 3) << avg_dist1, 0, -mean_pts1_x * avg_dist1,
            0, avg_dist1, -mean_pts1_y * avg_dist1,
            0, 0, 1);
    T2 = (cv::Mat_<float>(3, 3) << avg_dist2, 0, -mean_pts2_x * avg_dist2,
            0, avg_dist2, -mean_pts2_y * avg_dist2,
            0, 0, 1);

    auto *T1_ptr = (float *) T1.data;
    auto *T2_ptr = (float *) T2.data;

    norm_points = cv::Mat_<float>(sample_number, 4);

    auto *norm_points_ptr = (float *) norm_points.data;

    unsigned int norm_pts_idx;
    for (unsigned int i = 0; i < sample_number; i++) {
        smpl = 4 * sample[i];
        norm_pts_idx = 4 * i;
        norm_points_ptr[norm_pts_idx    ] = T1_ptr[0] * pts[smpl    ] + T1_ptr[2]; // Norm_img1_xi
        norm_points_ptr[norm_pts_idx + 1] = T1_ptr[4] * pts[smpl + 1] + T1_ptr[5]; // Norm_img1_yi

        norm_points_ptr[norm_pts_idx + 2] = T2_ptr[0] * pts[smpl + 2] + T2_ptr[2]; // Norm_img2_xi
        norm_points_ptr[norm_pts_idx + 3] = T2_ptr[4] * pts[smpl + 3] + T2_ptr[5]; // Norm_img2_yi
    }
}


void GetNormalizingTransformation (const float * const pts, cv::Mat& norm_points,
                                   const int * const sample, unsigned int sample_number, const float * const weightsx1, const float * const weightsy1,
                                   const float * const weightsx2, const float * const weightsy2, cv::Mat &T1, cv::Mat &T2) {

    float mean_pts1_x = 0, mean_pts1_y = 0, mean_pts2_x = 0, mean_pts2_y = 0;
    unsigned int smpl;
    unsigned int wsmpl;
    for (unsigned int i = 0; i < sample_number; i++) {
        wsmpl = sample[i];
        smpl = 4 * wsmpl;

        mean_pts1_x += weightsx1[wsmpl] * pts[smpl];
        mean_pts1_y += weightsy1[wsmpl] * pts[smpl+1];
        mean_pts2_x += weightsx2[wsmpl] * pts[smpl+2];
        mean_pts2_y += weightsy2[wsmpl] * pts[smpl+3];
    }
    mean_pts1_x /= sample_number;
    mean_pts1_y /= sample_number;
    mean_pts2_x /= sample_number;
    mean_pts2_y /= sample_number;

    float avg_dist1 = 0, avg_dist2 = 0, x1_m, y1_m, x2_m, y2_m;
    for (unsigned int i = 0; i < sample_number; i++) {
        wsmpl = sample[i];
        smpl = 4 * wsmpl;

        x1_m = weightsx1[wsmpl] * pts[smpl  ] - mean_pts1_x;
        y1_m = weightsy1[wsmpl] * pts[smpl+1] - mean_pts1_y;
        x2_m = weightsx2[wsmpl] * pts[smpl+2] - mean_pts2_x;
        y2_m = weightsy2[wsmpl] * pts[smpl+3] - mean_pts2_y;

        avg_dist1 += sqrt(x1_m * x1_m + y1_m * y1_m);
        avg_dist2 += sqrt(x2_m * x2_m + y2_m * y2_m);
    }

    // scale
    avg_dist1 = M_SQRT2 / (avg_dist1 / sample_number);
    avg_dist2 = M_SQRT2 / (avg_dist2 / sample_number);

    T1 = (cv::Mat_<float>(3, 3) << avg_dist1, 0, -mean_pts1_x * avg_dist1,
            0, avg_dist1, -mean_pts1_y * avg_dist1,
            0, 0, 1);
    T2 = (cv::Mat_<float>(3, 3) << avg_dist2, 0, -mean_pts2_x * avg_dist2,
            0, avg_dist2, -mean_pts2_y * avg_dist2,
            0, 0, 1);

    auto *T1_ptr = (float *) T1.data;
    auto *T2_ptr = (float *) T2.data;

    norm_points = cv::Mat_<float>(sample_number, 4);

    auto *norm_points_ptr = (float *) norm_points.data;

    unsigned int norm_pts_idx;
    for (unsigned int i = 0; i < sample_number; i++) {
        smpl = 4 * sample[i];
        norm_pts_idx = 4 * i;
        norm_points_ptr[norm_pts_idx    ] = T1_ptr[0] * pts[smpl    ] + T1_ptr[2]; // Norm_img1_xi
        norm_points_ptr[norm_pts_idx + 1] = T1_ptr[4] * pts[smpl + 1] + T1_ptr[5]; // Norm_img1_yi

        norm_points_ptr[norm_pts_idx + 2] = T2_ptr[0] * pts[smpl + 2] + T2_ptr[2]; // Norm_img2_xi
        norm_points_ptr[norm_pts_idx + 3] = T2_ptr[4] * pts[smpl + 3] + T2_ptr[5]; // Norm_img2_yi
    }
}
