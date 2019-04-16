// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef USAC_ESSENTIALESTIMATOR_H
#define USAC_ESSENTIALESTIMATOR_H

#include "estimator.hpp"
#include "fundamental/fundamental_solver.hpp"
#include "essential/five_points.hpp"

class EssentialEstimator : public Estimator {
private:
    const float * const points;
    float e11, e12, e13, e21, e22, e23, e31, e32, e33;
    EssentialSolver e_solver;
    FundamentalSolver f_solver;
    unsigned int points_size;
public:
    ~EssentialEstimator () override = default;

    /*
     * @input_points: is matrix of size: number of points x 4
     * x1 y1 x'1 y'1
     * ...
     * xN yN x'N y'N
     *
     * Y^T E Y = 0
     */
    EssentialEstimator(cv::InputArray input_points) : points((float *)input_points.getMat().data), 
                                                      e_solver (points), f_solver(points) {
        assert(!input_points.empty());
        points_size = input_points.getMat().rows;
    }

    void setModelParameters (const cv::Mat& model) override {
        float *  E_ptr = (float *) model.data;
        e11 = E_ptr[0]; e12 = E_ptr[1]; e13 = E_ptr[2];
        e21 = E_ptr[3]; e22 = E_ptr[4]; e23 = E_ptr[5];
        e31 = E_ptr[6]; e32 = E_ptr[7]; e33 = E_ptr[8];
    }

    /*
     * E = K1^T F K2
     *
     * y'^T E y = 0, normalized points by third coordinate.
     * x' = (y'1 y'2 y'3) / y'3
     * x  = (y1  y2  y3)  / y3
     */
    unsigned int EstimateModel(const int * const sample, std::vector<Model>& models) override {
        cv::Mat_<float> E;

        unsigned int models_count = e_solver.FivePoints (sample, E);

        for (unsigned int i = 0; i < models_count; i++) {
            models[i].setDescriptor(E.rowRange(i * 3, i * 3 + 3));
        }

        return models_count;
    }

    bool EstimateModelNonMinimalSample(const int * const sample, unsigned int sample_size, Model &model) override {
        cv::Mat_<float> E;

        if (! f_solver.EightPointsAlgorithm(sample, sample_size, E)) {
            return false;
        }

        model.setDescriptor(E);

        return true;
    }

    float GetError(unsigned int pidx) override {
        unsigned int smpl = 4*pidx;
        float x1 = points[smpl];
        float y1 = points[smpl+1];
        float x2 = points[smpl+2];
        float y2 = points[smpl+3];

        // pt2^T * E, line 1
        float l1 = x2 * e11 + y2 * e21 + e31;
        float l2 = x2 * e12 + y2 * e22 + e32;
        float l3 = x2 * e13 + y2 * e23 + e33;

        // E * pt1, line 2
        float t1 = e11 * x1 + e12 * y1 + e13;
        float t2 = e21 * x1 + e22 * y1 + e23;
        float t3 = e31 * x1 + e32 * y1 + e33;

        // distance from pt1 to line 1
        float a1 = l1 * x1 + l2 * y1 + l3;
        float a2 = sqrt(l1 * l1 + l2 * l2);

        // distance from pt2 to line 2
        float b1 = t1 * x2 + t2 * y2 + t3;
        float b2 = sqrt(t1 * t1 + t2 * t2);

        // get distances
        // distance1 = abs (a1 / a2)
        // distance2 = abs (b1 / b2)

        // error is average of distances
        return (fabsf(a1 / a2) + fabsf(b1 / b2)) / 2;
    }


    unsigned int getInliersWeights  (float threshold,
                                     int * inliers,
                                     bool get_error, float * errors,
                                     bool get_euc, float * weights_euc1,
                                     bool get_euc2, float * weights_euc2,
                                     bool sampson, float * weights_sampson,
                                     bool get_manh, float * weights_manh1,
                                     float * weights_manh2,
                                     float * weights_manh3,
                                     float * weights_manh4) override {
        float x1, y1, x2, y2, E_pt1_x, E_pt1_y, pt2_E_x, pt2_E_y, pt2_E_pt1, E_pt1_z, pt2_E_z;
        float err, dist_to2line, dist_to1line, sampson_err;
        unsigned int smpl, num_inliers = 0;
        for (unsigned int pt = 0; pt < points_size; pt++) {
            smpl = 4*pt;
            x1 = points[smpl];
            y1 = points[smpl+1];
            x2 = points[smpl+2];
            y2 = points[smpl+3];

            // line on second correspondence
            E_pt1_x = e11 * x1 + e12 * y1 + e13;
            E_pt1_y = e21 * x1 + e22 * y1 + e23;
            E_pt1_z = e31 * x1 + e32 * y1 + e33;

            // line on first correspondence
            pt2_E_x = x2 * e11 + y2 * e21 + e31;
            pt2_E_y = x2 * e12 + y2 * e22 + e32;
            pt2_E_z = x2 * e13 + y2 * e23 + e33;

            dist_to1line = fabsf(pt2_E_x * x1 + pt2_E_y * y1 + pt2_E_z) / sqrt(pt2_E_x * pt2_E_x + pt2_E_y * pt2_E_y);
            dist_to2line = fabsf(E_pt1_x * x2 + E_pt1_y * y2 + E_pt1_z) / sqrt(E_pt1_x * E_pt1_x + E_pt1_y * E_pt1_y);

            err = (dist_to1line + dist_to2line) / 2;

            if (err >= threshold) continue;

            inliers[num_inliers++] = pt;

            if (get_error) {
                errors[pt] = err;
            }

            if (sampson) {
                pt2_E_pt1 = x2 * E_pt1_x + y2 * E_pt1_y + e31 * x1 +  e32 * y1 + e33;
                sampson_err = (pt2_E_pt1 * pt2_E_pt1) / (E_pt1_x * E_pt1_x + E_pt1_y * E_pt1_y + pt2_E_x * pt2_E_x + pt2_E_y * pt2_E_y);
                weights_sampson[pt] = sampson_err < 1 ? 1 : 1 / sampson_err;
            }

            if (get_euc2) {
                weights_euc1[pt] = dist_to1line < 1 ? 1 : 1 / dist_to1line;
                weights_euc1[pt] = dist_to2line < 1 ? 1 : 1 / dist_to2line;
            }
        }
        return num_inliers;
    }

    static void getModelbyCameraMatrix (const cv::Mat &K1, const cv::Mat &K2, const cv::Mat &F, cv::Mat &E) {
        E =  K2.t() * F * K1;
    }

    int SampleNumber() override {
        return 5;
    }
};

#endif //USAC_ESSENTIALESTIMATOR_H
