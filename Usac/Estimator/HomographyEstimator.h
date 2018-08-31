#ifndef RANSAC_HOMOGRAPHYESTIMATOR_H
#define RANSAC_HOMOGRAPHYESTIMATOR_H


#include "Estimator.h"

class HomographyEstimator : public Estimator{
public:
    void EstimateModel(cv::InputArray input_points, int *sample, Model &model) override {
        const int idx1 = sample[0];
        const int idx2 = sample[1];
        const int idx3 = sample[2];
        const int idx4 = sample[3];

        cv::Point_<float> *points = (cv::Point_<float> *) input_points.getMat().data;

        // Estimate the model parameters from the sample
        float tangent_x = points[idx2].x - points[idx1].x;
        float tangent_y = points[idx2].y - points[idx1].y;

        float a = tangent_y;
        float b = -tangent_x;
        float mag = static_cast<float>(sqrt(a * a + b * b));
        a /= mag;
        b /= mag;
        float c = (points[idx2].x * points[idx1].y - points[idx2].y * points[idx1].x)/mag;

        // Set the model descriptor
        cv::Mat descriptor;
        descriptor = (cv::Mat_<float>(1,3) << a, b, c);
        model.setDescriptor(descriptor);
    }

    void EstimateModelNonMinimalSample(cv::InputArray input_points, int *sample, int sample_size, Model &model) override {
        cv::Mat points = cv::Mat(sample_size, 2, CV_32FC1);

        cv::Point_<float> *points_arr = (cv::Point_<float> *) input_points.getMat().data;

        for (int i = 0; i < sample_size; i++) {
            points.at<float>(i,0) = points_arr[sample[i]].x;
            points.at<float>(i,1) = points_arr[sample[i]].y;
        }

        cv::Mat covar, means, eigenvecs, eigenvals;;
        cv::calcCovarMatrix(points, covar, means, CV_COVAR_NORMAL | CV_COVAR_ROWS);
        cv::eigen (covar, eigenvals, eigenvecs);

        float a, b, c;
        a = (float) eigenvecs.at<double>(1,0);
        b = (float) eigenvecs.at<double>(1,1);
        c = (float) (-a*means.at<double>(0) - b*means.at<double>(1));

        cv::Mat descriptor;
        descriptor = (cv::Mat_<float>(1,3) << a, b, c);
        model.setDescriptor(descriptor);
    }

    float GetError(cv::InputArray input_points, int pidx, Model * model) override {
        cv::Mat descriptor;

        model->getDescriptor(descriptor);
        auto * params = reinterpret_cast<float *>(descriptor.data);

        return abs((int)(params[0] * input_points.getMat().at<float>(pidx,0) +
                         params[1] * input_points.getMat().at<float>(pidx,1) + params[2]));
    }

    int SampleNumber() override {
        return 4;
    }
};


#endif //RANSAC_HOMOGRAPHYESTIMATOR_H
