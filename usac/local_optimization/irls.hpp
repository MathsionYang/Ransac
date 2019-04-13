// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef USAC_IRLS_H
#define USAC_IRLS_H

#include "../model.hpp"
#include "../quality/quality.hpp"
#include "../estimator/dlt/dlt.hpp"
#include "../random_generator/uniform_random_generator.hpp"
#include <opencv2/plot.hpp>
#include "../../include/matplotlibcpp.h"

class Irls : public LocalOptimization {
private:
    Estimator * estimator;
    Quality * quality;
    Model irls_model;
    UniformRandomGenerator uniformRandomGenerator;
    RansacScore irls_score;

    float * weights;
    int *inliers, *sample;
    unsigned int points_size, max_sample_size;
    float threshold;

public:
    ~Irls() override {
        delete[] weights; delete[] inliers; delete[] sample;
    }

    Irls (Model * model, Estimator * estimator_, Quality * quality_, unsigned int points_size_) : irls_model (model){
        max_sample_size = model->lo_sample_size;
    
        weights = new float[points_size_];
        inliers = new int[points_size_];
        sample = new int [max_sample_size];

        quality = quality_;
        estimator = estimator_;
        points_size = points_size_;
        threshold = model->threshold;

        if (model->reset_random_generator) {
            uniformRandomGenerator.resetTime();
        }

    }

    void GetModelScore (Model * model, Score * score) override {
        irls_model.setDescriptor(model->returnDescriptor());
        Model std_lsq (model);
        RansacScore std_lsq_s;
        Model trimm_lsq (model);
        RansacScore trimm_lsq_s;
        Model irls_euc2 (model);
        RansacScore irls_euc2_s;
        Model irls_manh4 (model);
        RansacScore irls_manh4_s;

        float * weights_euc1 = new float [points_size];
        float * weights_euc2 = new float [points_size];
        float * weights_man1 = new float [points_size];
        float * weights_man2 = new float [points_size];
        float * weights_man3 = new float [points_size];
        float * weights_man4 = new float [points_size];

        unsigned int max_std_lsq_inl = 0;
        unsigned int max_trimm_lsq = 0;
        unsigned int max_euc1_inl = 0;
        unsigned int max_euc2_inl = 0;
        unsigned int max_manh4_inl = 0;

        float avg_std_lsq_inl = 0;
        float avg_trimm_lsq = 0;
        float avg_euc1_inl = 0;
        float avg_euc2_inl = 0;
        float avg_manh4_inl = 0;

        unsigned int num_samples;
        unsigned int num_iters = 20;
        for (unsigned int iter = 1; iter < num_iters; iter++) {
//            estimator->setModelParameters(irls_model.returnDescriptor());

            unsigned int num_inliers;

//            std::vector<float> errors;
////            cv::Mat data = cv::Mat_<float>(points_size, 1);
//            if (model->estimator == ESTIMATOR::Homography) {
//                estimator->getWeights(weights_euc1, weights_euc2, weights_man1, weights_man2, weights_man3, weights_man4);
//            } else {
//                estimator->GetError(weights, model->threshold, inliers, &num_inliers);
//            }


//            std::sort(errors.begin(), errors.end());
//            std::vector<float> y(errors.size());
//            std::iota(y.begin(), y.end(), 1);
//            matplotlibcpp::plot(y, errors, "r-");
//            matplotlibcpp::show();
//            matplotlibcpp::save("../results/1.png");
//            exit(0);

            //----------------------------------------------------
            estimator->setModelParameters(irls_model.returnDescriptor());
            estimator->getWeights(weights_euc1, weights_euc2, weights_man1, weights_man2, weights_man3, weights_man4);
            quality->getScore(&irls_score, irls_model.returnDescriptor(), irls_model.threshold, true, inliers);
            num_inliers = irls_score.inlier_number;
            num_samples = std::min(max_sample_size, num_inliers);
            uniformRandomGenerator.generateUniqueRandomSet(sample, num_samples, num_inliers-1);
            for (unsigned int smpl = 0; smpl < num_samples; smpl++) {
                sample[smpl] = inliers[sample[smpl]];
            }
            estimator->EstimateModelNonMinimalSample(sample, num_samples, weights_euc1, irls_model);
            std::cout << "irls 1 euclidean iteration " << iter << "; inliers size " << num_inliers << "\n";
            if (max_euc1_inl < num_inliers) max_euc1_inl = num_inliers;
            avg_euc1_inl += num_inliers;

            //----------------------------------------------------
            estimator->setModelParameters(irls_euc2.returnDescriptor());
            estimator->getWeights(weights_euc1, weights_euc2, weights_man1, weights_man2, weights_man3, weights_man4);
            quality->getScore(&irls_euc2_s, irls_euc2.returnDescriptor(), irls_euc2.threshold, true, inliers);
            num_inliers = irls_euc2_s.inlier_number;
            num_samples = std::min(max_sample_size, num_inliers);
            uniformRandomGenerator.generateUniqueRandomSet(sample, num_samples, num_inliers-1);
            for (unsigned int smpl = 0; smpl < num_samples; smpl++) {
                sample[smpl] = inliers[sample[smpl]];
            }
            estimator->EstimateModelNonMinimalSample(sample, num_samples, weights_euc1, weights_euc2, irls_euc2);
            std::cout << "irls 2 euclidean iteration " << iter << "; inliers size " << num_inliers << "\n";
            if (max_euc2_inl < num_inliers) max_euc2_inl = num_inliers;
            avg_euc2_inl += num_inliers;

            //----------------------------------------------------
            estimator->setModelParameters(irls_manh4.returnDescriptor());
            estimator->getWeights(weights_euc1, weights_euc2, weights_man1, weights_man2, weights_man3, weights_man4);
            quality->getScore(&irls_manh4_s, irls_manh4.returnDescriptor(), irls_manh4.threshold, true, inliers);
            num_inliers = irls_manh4_s.inlier_number;
            num_samples = std::min(max_sample_size, num_inliers);
            uniformRandomGenerator.generateUniqueRandomSet(sample, num_samples, num_inliers-1);
            for (unsigned int smpl = 0; smpl < num_samples; smpl++) {
                sample[smpl] = inliers[sample[smpl]];
            }
            estimator->EstimateModelNonMinimalSample(sample, num_samples, weights_man1, weights_man2, weights_man3, weights_man4, irls_manh4);
            std::cout << "irls 4 manhattan iteration " << iter << "; inliers size " << num_inliers << "\n";
            if (max_manh4_inl < num_inliers) max_manh4_inl = num_inliers;
            avg_manh4_inl += num_inliers;

            //----------------------------------------------------
            estimator->setModelParameters(std_lsq.returnDescriptor());
            quality->getScore(&std_lsq_s, std_lsq.returnDescriptor(), std_lsq.threshold, true, inliers);
            num_inliers = std_lsq_s.inlier_number;
            num_samples = std::min(max_sample_size, num_inliers);
            uniformRandomGenerator.generateUniqueRandomSet(sample, num_samples, num_inliers-1);
            for (unsigned int smpl = 0; smpl < num_samples; smpl++) {
                sample[smpl] = inliers[sample[smpl]];
            }
            estimator->EstimateModelNonMinimalSample(sample, num_samples, std_lsq);
            std::cout << "standard least squares iteration " << iter << "; inliers size " << num_inliers << "\n";
            if (max_std_lsq_inl < num_inliers) max_std_lsq_inl = num_inliers;
            avg_std_lsq_inl += num_inliers;

            //----------------------------------------------------
            estimator->setModelParameters(trimm_lsq.returnDescriptor());
            quality->getScore(&trimm_lsq_s, trimm_lsq.returnDescriptor(), trimm_lsq.threshold, true, inliers);
            num_inliers = trimm_lsq_s.inlier_number;
            num_samples = std::min(max_sample_size, num_inliers);
            std::sort(inliers, inliers+num_inliers);
            estimator->EstimateModelNonMinimalSample(inliers, num_samples, trimm_lsq);
            std::cout << "trimmed least squares iteration " << iter << "; inliers size " << num_inliers << "\n";
            if (max_trimm_lsq < num_inliers) max_trimm_lsq = num_inliers;
            avg_trimm_lsq += num_inliers;

            std::cout << "============================================================================================\n";
//            if (num_inliers > score->inlier_number) {
//                std::cout << "UPDATE SCORE\n";
//                score->inlier_number = num_inliers;
//                score->score = num_inliers;
//                model->setDescriptor(irls_model.returnDescriptor());
//            }
        }

        std::cout << "max euc1 " << max_euc1_inl << "\n";
        std::cout << "max euc2 " << max_euc2_inl << "\n";
        std::cout << "max manh4 " << max_manh4_inl << "\n";
        std::cout << "max std lsq " << max_std_lsq_inl << "\n";
        std::cout << "max trimm lsq " << max_trimm_lsq << "\n";
        std::cout << "- - - - \n";
        std::cout << "avg euc1 " << avg_euc1_inl / num_iters << "\n";
        std::cout << "avg euc2 " << avg_euc2_inl / num_iters << "\n";
        std::cout << "avg manh4 " << avg_manh4_inl / num_iters << "\n";
        std::cout << "avg std lsq " << avg_std_lsq_inl / num_iters << "\n";
        std::cout << "avg trimm lsq " << avg_trimm_lsq / num_iters << "\n";


        // end:

//        quality->getScore(&irls_score, irls_model.returnDescriptor());
//        std::cout << "irls iteration 20; inliers size " << irls_score.inlier_number << "\n";

//        if (irls_score.bigger(score)) {
//            std::cout << "UPDATE SCORE\n";
//            score->copyFrom(irls_score);
//            model->setDescriptor(irls_model.returnDescriptor());
//        }
//        std::cout << "-----------------------------------------------\n";
    }

    unsigned int getNumberIterations () override {
        return 0;
    }
};

#endif //USAC_IRLS_H
