// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef RANSAC_HOMOGRAPHYESTIMATOR_H
#define RANSAC_HOMOGRAPHYESTIMATOR_H

#include "estimator.hpp"
#include "dlt/dlt.hpp"

class HomographyEstimator : public Estimator{
private:
    const float * const points;
    float h11, h12, h13, h21, h22, h23, h31, h32, h33;
    DLt * dlt;
    unsigned int points_size;
public:
    ~HomographyEstimator () {
        delete (dlt);
    }

    /*
     * @input_points: is matrix of size: number of points x 4
     * x1 y1 x'1 y'1
     * ...
     * xN yN x'N y'N
     */
    HomographyEstimator(cv::InputArray input_points) : points((float *)input_points.getMat().data) {
        assert(!input_points.empty());
        dlt = new DLt(points);
        points_size = input_points.getMat().rows;
    }

    void setModelParameters (const cv::Mat& model) override {
        auto * H_ptr = (float *) model.data;

        h11 = H_ptr[0]; h12 = H_ptr[1]; h13 = H_ptr[2];
        h21 = H_ptr[3]; h22 = H_ptr[4]; h23 = H_ptr[5];
        h31 = H_ptr[6]; h32 = H_ptr[7]; h33 = H_ptr[8];
    }

    unsigned int EstimateModel(const int * const sample, std::vector<Model*>& models) override {
        cv::Mat H;
        if (! dlt->DLT4p (sample, H)) {
            return 0;
        }

        models[0]->setDescriptor(H);

        return 1;
    }

    bool EstimateModelNonMinimalSample (const int * const sample, unsigned int sample_size, Model &model) override {
        cv::Mat H;
        if (! dlt->NormalizedDLT(sample, sample_size, H)) {
            return false;
        }

        model.setDescriptor(H);
        return true;
    }

    bool EstimateModelNonMinimalSample (const int * const sample, unsigned int sample_size, const float * const weights, Model &model) override {
        cv::Mat H;
        if (! dlt->NormalizedDLT(sample, sample_size, weights, H)) {
            return false;
        }

        model.setDescriptor(H);
        return true;
    }

    bool LeastSquaresFitting (const int * const sample, unsigned int sample_size, Model &model) override {
        return EstimateModelNonMinimalSample(sample, sample_size, model);
    }
    /*
     * Error = distance (H pt(i), pt'(i))
     */
    float GetError(unsigned int pidx) override {
        unsigned int smpl = 4*pidx;
        float x1 = points[smpl];
        float y1 = points[smpl+1];
        float x2 = points[smpl+2];
        float y2 = points[smpl+3];

        float est_x2 = h11 * x1 + h12 * y1 + h13;
        float est_y2 = h21 * x1 + h22 * y1 + h23;
        float est_z2 = h31 * x1 + h32 * y1 + h33; // h33 = 1

        est_x2 /= est_z2;
        est_y2 /= est_z2;
        
        float error = sqrt ((x2 - est_x2) * (x2 - est_x2) + (y2 - est_y2) * (y2 - est_y2));
        // error >= 0
        return error;
    }

    float GetTotalError (float threshold, bool get_inliers, int * inliers) override {
        float x1, y1, x2, y2, est_x2, est_y2, est_z2, error = 0;
        unsigned int smpl;

        for (unsigned int pt = 0; pt < points_size; pt++) {
            smpl = 4*pt;
            x1 = points[smpl];
            y1 = points[smpl+1];
            x2 = points[smpl+2];
            y2 = points[smpl+3];

            est_x2 = h11 * x1 + h12 * y1 + h13;
            est_y2 = h21 * x1 + h22 * y1 + h23;
            est_z2 = h31 * x1 + h32 * y1 + h33; // h33 = 1

            est_x2 /= est_z2;
            est_y2 /= est_z2;
            
            error += sqrt ((x2 - est_x2) * (x2 - est_x2) + (y2 - est_y2) * (y2 - est_y2));
        }

        return error;
    }


    int SampleNumber() override {
        return 4;
    }
};


#endif //RANSAC_HOMOGRAPHYESTIMATOR_H