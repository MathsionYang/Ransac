#ifndef USAC_FUNDAMENTALESTIMATOR_H
#define USAC_FUNDAMENTALESTIMATOR_H

#include "estimator.hpp"
#include "fundamental/fundamental_solver.hpp"

class FundamentalEstimator : public Estimator {
private:
    const float * const points;
    float f11, f12, f13, f21, f22, f23, f31, f32, f33;
    unsigned int points_size;
    FundamentalSolver solver;
public:
    ~FundamentalEstimator () override = default;
    /*
     * @input_points: is matrix of size: number of points x 4
     * x1 y1 x'1 y'1
     * ...
     * xN yN x'N y'N
     *
     * X^T F X = 0
     */
    FundamentalEstimator(cv::InputArray input_points) : points((float *)input_points.getMat().data), solver (points) {
        assert(!input_points.empty());
        points_size = input_points.getMat().rows;
    }

    void setModelParameters (const cv::Mat& model) override {

        /*
         * To make pointer from Mat class, this Mat class should exists as long as exists pointer
         * So this->F and this->F_inv must be global in class
         */
        auto * F_ptr = (float *) model.data;
        f11 = F_ptr[0]; f12 = F_ptr[1]; f13 = F_ptr[2];
        f21 = F_ptr[3]; f22 = F_ptr[4]; f23 = F_ptr[5];
        f31 = F_ptr[6]; f32 = F_ptr[7]; f33 = F_ptr[8];
    }

    unsigned int EstimateModel(const int * const sample, std::vector<Model>& models) override {
        cv::Mat_<float> F;

        unsigned int roots = solver.SevenPointsAlgorithm(sample, F);

//        std::cout << "Roots " << roots << "\n\n";

        unsigned int valid_solutions = 0;
        for (unsigned int i = 0; i < roots; i++) {
            if (isModelValid(F.rowRange(i * 3, i * 3 + 3), sample)) {
                models[valid_solutions++].setDescriptor(F.rowRange(i * 3, i * 3 + 3));
            }
        }

        return valid_solutions;
    }

    bool EstimateModelNonMinimalSample(const int * const sample, unsigned int sample_size, Model &model) override {
        cv::Mat_<float> F;

        if (! solver.EightPointsAlgorithm(sample, sample_size, F)) {
                return false;
        }

        model.setDescriptor(F);

        return true;
    }

    bool EstimateModelNonMinimalSample(const int * const sample, unsigned int sample_size, const float *const weights, Model &model) override {
        cv::Mat_<float> F;

        if (! solver.EightPointsAlgorithm(sample, weights, sample_size, F)) {
                return false;
        }

        model.setDescriptor(F);

        return true;
    }

    /*
     * Sampson error
     *                               (pt2^t * F * pt1)^2)
     * Error =  -------------------------------------------------------------------
     *          (((F⋅pt1)(0))^2 + ((F⋅pt1)(1))^2 + ((F^t⋅pt2)(0))^2 + ((F^t⋅pt2)(1))^2)
     *
     * [ x2 y2 1 ] * [ F(1,1)  F(1,2)  F(1,3) ]   [ x1 ]
     *               [ F(2,1)  F(2,2)  F(2,3) ] * [ y1 ]
     *               [ F(3,1)  F(3,2)  F(3,3) ]   [ 1  ]
     *
     */
    float GetError(unsigned int pidx) override {
        unsigned int smpl = 4*pidx;
        float x1 = points[smpl];
        float y1 = points[smpl+1];
        float x2 = points[smpl+2];
        float y2 = points[smpl+3];

        float F_pt1_x = f11 * x1 + f12 * y1 + f13;
        float F_pt1_y = f21 * x1 + f22 * y1 + f23;

        float pt2_F_x = x2 * f11 + y2 * f21 + f31;
        float pt2_F_y = x2 * f12 + y2 * f22 + f32;

        float pt2_F_pt1 = x2 * F_pt1_x + y2 * F_pt1_y + f31 * x1 +  f32 * y1 + f33; // f33 = 1

        float error = (pt2_F_pt1 * pt2_F_pt1) / (F_pt1_x * F_pt1_x + F_pt1_y * F_pt1_y + pt2_F_x * pt2_F_x + pt2_F_y * pt2_F_y);

        // debug
//        cv::Mat pt1 = (cv::Mat_<double>(3,1) << x1, y1, 1);
//        cv::Mat pt2 = (cv::Mat_<double>(3,1) << x2, y2, 1);
//        cv::Mat F_double = (cv::Mat_<double>(3,3) << f11, f12, f13, f21, f22, f23, f31, f32, f33);
//        double error_opencv = cv::sampsonDistance(pt1, pt2, F_double);
//        if (fabsf (error - (float) error_opencv) > 1) {
//            std::cout << "error " << error << " VS opencv error " << error_opencv << "\n";
//            std::cout << "difference " << fabsf (error - (float) error_opencv) << "\n";
//            std::cout << "Check GetError in Fundamental Matrix Estimator!\n";
//        }
        //

        // std::cout << "error = " << error << '\n';
        // error >= 0
        return error;
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

        float x1, y1, x2, y2, F_pt1_x, F_pt1_y, pt2_F_x, pt2_F_y, pt2_F_pt1, F_pt1_z, pt2_F_z;
        float err, dist_to2line, dist_to1line;
        unsigned int smpl, num_inliers = 0;
        for (unsigned int pt = 0; pt < points_size; pt++) {
            smpl = 4*pt;
            x1 = points[smpl];
            y1 = points[smpl+1];
            x2 = points[smpl+2];
            y2 = points[smpl+3];

            // line on 2 correspondence
            F_pt1_x = f11 * x1 + f12 * y1 + f13;
            F_pt1_y = f21 * x1 + f22 * y1 + f23;

            // line on 1 correspondence
            pt2_F_x = x2 * f11 + y2 * f21 + f31;
            pt2_F_y = x2 * f12 + y2 * f22 + f32;

            pt2_F_pt1 = x2 * F_pt1_x + y2 * F_pt1_y + f31 * x1 +  f32 * y1 + f33; // f33 = 1

            err = (pt2_F_pt1 * pt2_F_pt1) / (F_pt1_x * F_pt1_x + F_pt1_y * F_pt1_y + pt2_F_x * pt2_F_x + pt2_F_y * pt2_F_y);

            if (err >= threshold) continue;
            inliers[num_inliers++] = pt;

            if (get_error) {
                errors[pt] = err;
            }

            if (sampson) {
                weights_sampson[pt] = err < 1 ? 1 : 1 / err;
            }

            if (get_euc2) {
                F_pt1_z = f31 * x1 + f32 * y1 + f33;
                pt2_F_z = f13 * x1 + f23 * y1 + f33;

                dist_to1line = (pt2_F_x * x1 + pt2_F_y * y1 + pt2_F_z) / sqrt(pt2_F_x * pt2_F_x + pt2_F_y * pt2_F_y);
                dist_to2line = (F_pt1_x * x2 + F_pt1_y * y2 + F_pt1_z) / sqrt(F_pt1_x * F_pt1_x + F_pt1_y * F_pt1_y);

                weights_euc1[pt] = dist_to1line < 1 ? 1 : 1 / dist_to1line;
                weights_euc2[pt] = dist_to2line < 1 ? 1 : 1 / dist_to2line;
            }

        }
        return num_inliers;
    }


    static void getModelbyCameraMatrix (const cv::Mat &K1, const cv::Mat &K2, const cv::Mat &E, cv::Mat &F) {
        F =  K2.inv().t() * E * K1.inv();
    }

    int SampleNumber() override {
        return 7;
    }


    static void getFundamentalFromProjection(const cv::Mat &P1, const cv::Mat &P2, cv::Mat &F) {
        cv::SVD svd(P1, 4);

        // e1 = svd.vt.row(3)
        cv::Mat e2 = P2 * svd.vt.row(3).t();

        cv::Mat e2x = (cv::Mat_<float>(3, 3) << 0, -e2.at<float>(2), e2.at<float>(1),
                                                e2.at<float>(2), 0, -e2.at<float>(0),
                                               -e2.at<float>(1), e2.at<float>(0), 0);

        F = e2x * P2 * P1.inv(cv::DECOMP_SVD);
        F = F / F.at<float>(2, 2);
    }


    bool isModelValid(const cv::Mat &F, const int * const sample) override {
        cv::Mat ec;
        float sig, sig1;
        int i;
        epipole(ec, F);

        sig1 = getorisig(F, &ec, 4*sample[0]);

        for (i = 1; i < 7; i++) {
            sig = getorisig(F, &ec, 4*sample[i]);

            if (sig1 * sig < 0) return false;
        }
        return true;
    }

private:
    // https://github.com/danini/graph-cut-ransac/blob/master/GraphCutRANSAC/essential_estimator.cpp
    /************** oriented constraints ******************/
    void epipole(cv::Mat &ec, const cv::Mat &F) const {
        ec = F.row(0).cross(F.row(2));

        for (int i = 0; i < 3; i++)
            if ((ec.at<float>(i) > 1.9984e-15) || (ec.at<float>(i) < -1.9984e-15)) return;
        ec = F.row(1).cross(F.row(2));
    }

    // u = x1 y1 1 x2 y2 1
    float getorisig(const cv::Mat &F, const cv::Mat *ec, unsigned int pt_idx) const {
        float s1, s2;

        float y1 = points[pt_idx+1];
        float x2 = points[pt_idx+2];
        float y2 = points[pt_idx+3];

//        s1 = F->at<float>(0) * u.at<float>(3) + F->at<float>(3) * u.at<float>(4) + F->at<float>(6) * u.at<float>(5);
//        s2 = ec->at<float>(1) * u.at<float>(2) - ec->at<float>(2) * u.at<float>(1);

        s1 = F.at<float>(0) * x2 + F.at<float>(3) * y2 + F.at<float>(6);
        s2 = ec->at<float>(1) - ec->at<float>(2) * y1;

        return(s1 * s2);
    }
};

#endif //USAC_FUNDAMENTALESTIMATOR_H