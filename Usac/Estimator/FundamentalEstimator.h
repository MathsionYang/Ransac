#ifndef USAC_FUNDAMENTALESTIMATOR_H
#define USAC_FUNDAMENTALESTIMATOR_H

#include "Estimator.h"
#include "NPointsAlgorithms/EightPointsAlgorithm.h"

class FundamentalEstimator : public Estimator {
private:
    const float * const points;
    cv::Mat F;
    float *F_ptr;
public:

    /*
     * input_points must be:
     * img1_x1 img1_y1 img2_x1 img2_y1
     * img1_x2 img1_y2 img2_x2 img2_y2
     * ....
     * img1_xN img1_yN img2_xN img2_yN
     *
     * Size N x (2*|imgs|)
     *
     *
     * float array 4N x 1
     * img1_x1
     * img1_y1
     * img2_x1
     * img2_y1
     * img1_x2
     * img1_y2
     * img2_x2
     * img2_y2
     * ...
     * img1_xN
     * img1_yN
     * img2_xN
     * img2_yN
     */

    FundamentalEstimator(cv::InputArray input_points) : points((float *)input_points.getMat().data) {
        assert(!input_points.empty());
    }

    void setModelParameters (Model * const model) override {
        F = cv::Mat_<float>(model->returnDescriptor());

        /*
         * Attention!
         * To make pointer from Mat class, this Mat class should exists as long as exists pointer
         * So this->F and this->F_inv must be global in class
         */
        F_ptr = (float *) F.data;
    }

    int EstimateModel(const int * const sample, Model ** models) override {
        cv::Mat_<float> F;

        int roots = SevenPointsAlgorithm(points, sample, F);
        if (roots < 1) {
            std::cout << "roots less than 1\n";
            return 0;
        }

        models[0]->setDescriptor(F.rowRange(0,3));
//        std::cout << F << "\n\n";

        if (roots > 1) {
            models = (Model **) (realloc(models, roots * sizeof(Model *)));
            for (int i = 1; i < roots; i++) {
                models[i] = new Model(models[0]->threshold,
                                      models[0]->sample_number,
                                      models[0]->desired_prob,
                                      models[0]->k_nearest_neighbors,
                                      models[0]->model_name);

                models[i]->setDescriptor(F.rowRange(i * 3, i * 3 + 3));
            }
        }

        return roots;
    }

    void EstimateModelNonMinimalSample(const int * const sample, int sample_size, Model &model) override {
        cv::Mat_<float> F;
        EightPointsAlgorithm(points, sample, sample_size, F);

        model.setDescriptor(F);
    }

    /*
     * Sampson error
     *                               (pt2^t * F * pt1)^2)
     * Error =  -------------------------------------------------------------------
     *          (((F⋅pt1)(0))^2 + ((F⋅pt1)(1))^2 + ((F^t⋅pt2)(0))^2 + ((F^t⋅pt2)(1))^2)
     *
     * ( [ x2 y2 1 ] * [ F(1,1)  F(1,2)  F(1,3) ] )   [ x1 ]
     * (               [ F(2,1)  F(2,2)  F(2,3) ] ) * [ y1 ]
     * (               [ F(3,1)  F(3,2)  F(3,3) ] )   [ 1  ]
     *
     */
    float GetError(int pidx) override {
        unsigned int smpl = 4*pidx;
        float x1 = points[smpl];
        float y1 = points[smpl+1];
        float x2 = points[smpl+2];
        float y2 = points[smpl+3];


        float F_pt1_x = F_ptr[0] * x1 + F_ptr[1] * y1 + F_ptr[2];
        float F_pt1_y = F_ptr[3] * x1 + F_ptr[4] * y1 + F_ptr[5];

        // Here F is transposed
        float F_pt2_x = F_ptr[0] * x2 + F_ptr[3] * y2 + F_ptr[6];
        float F_pt2_y = F_ptr[1] * x2 + F_ptr[4] * y2 + F_ptr[7];

        float pt2_F_pt1 = x2 * F_pt1_x + y2 * F_pt1_y + F_ptr[6] * x1 +  F_ptr[7] * y1 +  F_ptr[8];

        float error = (pt2_F_pt1 * pt2_F_pt1) / (F_pt1_x * F_pt1_x + F_pt1_y * F_pt1_y + F_pt2_x * F_pt2_x + F_pt2_y * F_pt2_y);
//        std::cout << "error = " << error << '\n';

        return error;
    }

    int SampleNumber() override {
        return 7;
    }
};

#endif //USAC_FUNDAMENTALESTIMATOR_H