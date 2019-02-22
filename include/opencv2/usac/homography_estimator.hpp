// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef RANSAC_HOMOGRAPHYESTIMATOR_H
#define RANSAC_HOMOGRAPHYESTIMATOR_H

#include "estimator.hpp"
#include "dlt.hpp"

namespace cv { namespace usac {
class HomographyEstimator : public Estimator {
private:
    const float *const points;
    float h11, h12, h13, h21, h22, h23, h31, h32, h33;
    DLt dlt;
public:
    /*
     * @input_points: is matrix of size: number of points x 4
     * x1 y1 x'1 y'1
     * ...
     * xN yN x'N y'N
     */
    HomographyEstimator(cv::InputArray input_points) : points((float *) input_points.getMat().data), dlt (points){
        assert(!input_points.empty());
        assert(input_points.getMat().cols == 4 && input_points.getMat().rows >= 4);
    }

    void setModelParameters(const cv::Mat &model) override {
        auto *H_ptr = (float *) model.data;

        h11 = H_ptr[0]; h12 = H_ptr[1]; h13 = H_ptr[2];
        h21 = H_ptr[3]; h22 = H_ptr[4]; h23 = H_ptr[5];
        h31 = H_ptr[6]; h32 = H_ptr[7]; h33 = H_ptr[8];
    }

    unsigned int estimateModel(const int *const sample, std::vector<Model *> &models) override {
        cv::Mat H;
        if (!dlt.DLT4p(sample, H)) {
            return 0;
        }

        models[0]->setDescriptor(H);

        return 1;
    }

    bool
    estimateModelNonMinimalSample(const int *const sample, unsigned int sample_size, Model &model) override {
        cv::Mat H;
        if (!dlt.NormalizedDLT(sample, sample_size, H)) {
//            std::cout << "Normalized DLT failed\n";
            return false;
        }

        model.setDescriptor(H);
        return true;
    }

    bool leastSquaresFitting(const int *const sample, unsigned int sample_size, Model &model) override {
        return estimateModelNonMinimalSample(sample, sample_size, model);
    }

    /*
     * Error = mean (distance (pt(i)H, pt'(i)) + distance (pt(i), pt'(i)H^-1))
     */
    float getError(unsigned int pidx) override {
        unsigned int smpl = 4 * pidx;
        float x1 = points[smpl];
        float y1 = points[smpl + 1];
        float x2 = points[smpl + 2];
        float y2 = points[smpl + 3];

        float est_x2 = h11 * x1 + h12 * y1 + h13;
        float est_y2 = h21 * x1 + h22 * y1 + h23;
        float est_z2 = h31 * x1 + h32 * y1 + h33; // h33 = 1

        est_x2 /= est_z2;
        est_y2 /= est_z2;

        float error = sqrt((x2 - est_x2) * (x2 - est_x2) + (y2 - est_y2) * (y2 - est_y2));

        return error;
    }

    int sampleNumber() override {
        return 4;
    }
};
}}

#endif //RANSAC_HOMOGRAPHYESTIMATOR_H