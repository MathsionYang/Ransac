#include "dlt.hpp"

bool DLt::NormalizedDLT (const int * const sample, unsigned int sample_number, cv::Mat &H) {
    cv::Mat T1, T2, norm_points;
    GetNormalizingTransformation(points, norm_points, sample, sample_number, T1, T2);

    const float * const norm_points_ptr = (float *) norm_points.data;

    if (! DLT(norm_points_ptr, sample_number, H)) {
//    if (! DLTEigen(norm_points_ptr, sample_number, H)) {
            return false;
    }

    H = T2.inv()*H*T1;
    // normalize H by last h33
    H = H / H.at<float>(2,2);

    return true;
}

// Weighted Normalized DLT, weights go to Normalizing Transformation
bool DLt::NormalizedDLT (const int * const sample, unsigned int sample_number, const float * const weights, cv::Mat &H) {
    cv::Mat T1, T2, norm_points;
    GetNormalizingTransformation(points, norm_points, sample, sample_number, weights, T1, T2);
    const float * const norm_points_ptr = (float *) norm_points.data;
    if (! DLT(norm_points_ptr, sample_number, H)) {
        return false;
    }
    H = T2.inv()*H*T1;
    H = H / H.at<float>(2,2);
    return true;
}

bool DLt::NormalizedDLT (const int * const sample, unsigned int sample_number, const float * const weightsx, const float * const weightsy, cv::Mat &H) {
    cv::Mat T1, T2, norm_points;
    GetNormalizingTransformation(points, norm_points, sample, sample_number, weightsx, weightsy, T1, T2);
    const float * const norm_points_ptr = (float *) norm_points.data;
    if (! DLT(norm_points_ptr, sample_number, H)) {
        return false;
    }
    H = T2.inv()*H*T1;
    H = H / H.at<float>(2,2);
    return true;
}

bool DLt::NormalizedDLT (const int * const sample, unsigned int sample_number, const float * const weightsx1, const float * const weightsy1,
                            const float * const weightsx2, const float * const weightsy2, cv::Mat &H) {
    cv::Mat T1, T2, norm_points;
    GetNormalizingTransformation(points, norm_points, sample, sample_number, weightsx1, weightsy1, weightsx2, weightsy2, T1, T2);
    const float * const norm_points_ptr = (float *) norm_points.data;
    if (! DLT(norm_points_ptr, sample_number, H)) {
        return false;
    }
    H = T2.inv()*H*T1;
    H = H / H.at<float>(2,2);
    return true;
}
