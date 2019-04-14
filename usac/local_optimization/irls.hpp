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

    float *errors, *weights_euc1, *weights_euc2, *weights_manh1, *weights_manh2, *weights_manh3, *weights_manh4;
    int *inliers, *sample;
    unsigned int points_size, max_sample_size, num_lo_iters;
    float threshold;

public:
    ~Irls() override {
        delete[] inliers; delete[] sample;
    }

    Irls (Model * model, Estimator * estimator_, Quality * quality_, unsigned int points_size_) : irls_model (model){
        max_sample_size = model->lo_sample_size;
        points_size = points_size_;

        errors = new float[points_size];
        weights_euc1 = new float [points_size];
        weights_euc2 = new float [points_size];
        weights_manh1 = new float [points_size];
        weights_manh2 = new float [points_size];
        weights_manh3 = new float [points_size];
        weights_manh4 = new float [points_size];

        inliers = new int[points_size_];
        sample = new int [max_sample_size];

        quality = quality_;
        estimator = estimator_;
        threshold = model->threshold;
        num_lo_iters = 20; // model->lo_inner_iterations;

        if (model->reset_random_generator) {
            uniformRandomGenerator.resetTime();
        }
    }

    void GetModelScore (Model * model, Score * score) override {
//            std::sort(errors.begin(), errors.end());
//            std::vector<float> y(errors.size());
//            std::iota(y.begin(), y.end(), 1);
//            matplotlibcpp::plot(y, errors, "r-");
//            matplotlibcpp::show();
//            matplotlibcpp::save("../results/1.png");
//            exit(0);

        runIRLSEuclideanOneWeights(model, score);
        runIRLSEuclideanTwoWeights(model, score);
        runIRLSManhattanFourWeights(model, score);
        runIRLSIdentityWeights(model, score);
//        runIRLSIdentityWeights(model, score, true);
    }

    void runIRLSEuclideanOneWeights (Model * model, Score * score, bool trimmed=false) {
        irls_model.setDescriptor(model->returnDescriptor());

        unsigned int max_score = 0;
        unsigned int min_score = 10000000;
        float avg_score = 0;
        unsigned int num_samples, num_inliers;
        for (unsigned int iter = 1; iter < num_lo_iters; iter++) {
            estimator->setModelParameters(irls_model.returnDescriptor());

            if (trimmed) {
                num_inliers = estimator->getInliersWeights(irls_model.threshold, inliers,
                                                           true, errors, true, weights_euc1, false, nullptr, false, nullptr, nullptr, nullptr, nullptr);
                num_samples = std::min(max_sample_size, num_inliers);
                if (num_samples <= model->sample_size) continue;
                std::sort (inliers, inliers+num_inliers, [&](int a, int b) {
                    return errors[a] < errors[b];
                });
                estimator->EstimateModelNonMinimalSample(inliers, num_samples, weights_euc1, irls_model);
            } else {
                num_inliers = estimator->getInliersWeights(irls_model.threshold, inliers,
                                                           false, nullptr, true, weights_euc1, false, nullptr, false, nullptr, nullptr, nullptr, nullptr);
                num_samples = std::min(max_sample_size, num_inliers);
                if (num_samples <= model->sample_size) continue;
                uniformRandomGenerator.generateUniqueRandomSet(sample, num_samples, num_inliers - 1);
                for (unsigned int smpl = 0; smpl < num_samples; smpl++) {
                    sample[smpl] = inliers[sample[smpl]];
                }
                estimator->EstimateModelNonMinimalSample(sample, num_samples, weights_euc1, irls_model);
            }
            irls_score.inlier_number = num_inliers;
            irls_score.score = irls_score.inlier_number;

//            if (irls_score.bigger(score)) {
//                score->copyFrom(irls_score);
//                model->setDescriptor(irls_model.returnDescriptor());
//            }

            std::cout << "irls trimmed " << trimmed << " one euclidean iteration " << iter << "; inliers size " << num_inliers << "\n";
            if (iter != 0 && max_score < num_inliers) max_score = num_inliers;
            if (iter != 0 && min_score > num_inliers) min_score = num_inliers;
            avg_score += num_inliers;
        }
        std::cout << "max score " << max_score << "\n";
        std::cout << "min score " << min_score << "\n";
        std::cout << "avg score " << avg_score / num_lo_iters << "\n";
        std::cout << "-----------------------------------------------\n";
    }

    void runIRLSEuclideanTwoWeights (Model * model, Score * score, bool trimmed=false) {
        irls_model.setDescriptor(model->returnDescriptor());

        unsigned int max_score = 0;
        unsigned int min_score = 10000000;
        float avg_score = 0;
        unsigned int num_samples, num_inliers;
        for (unsigned int iter = 1; iter < num_lo_iters; iter++) {
            estimator->setModelParameters(irls_model.returnDescriptor());

            if (trimmed) {
                num_inliers = estimator->getInliersWeights(irls_model.threshold, inliers,
                                                           true, errors, true, weights_euc1, true, weights_euc2, false, nullptr, nullptr, nullptr, nullptr);
                num_samples = std::min(max_sample_size, num_inliers);
                if (num_samples <= model->sample_size) continue;
                std::sort (inliers, inliers+num_inliers, [&](int a, int b) {
                    return errors[a] < errors[b];
                });
                estimator->EstimateModelNonMinimalSample(inliers, num_samples, weights_euc1, weights_euc2, irls_model);
            } else {
                num_inliers = estimator->getInliersWeights(irls_model.threshold, inliers,
                                                           false, nullptr, true, weights_euc1, true, weights_euc2, false, nullptr, nullptr, nullptr, nullptr);
                num_samples = std::min(max_sample_size, num_inliers);
                if (num_samples <= model->sample_size) continue;
                uniformRandomGenerator.generateUniqueRandomSet(sample, num_samples, num_inliers - 1);
                for (unsigned int smpl = 0; smpl < num_samples; smpl++) {
                    sample[smpl] = inliers[sample[smpl]];
                }
                estimator->EstimateModelNonMinimalSample(sample, num_samples, weights_euc1, weights_euc2, irls_model);
            }
            irls_score.inlier_number = num_inliers;
            irls_score.score = irls_score.inlier_number;

//            if (irls_score.bigger(score)) {
//                score->copyFrom(irls_score);
//                model->setDescriptor(irls_model.returnDescriptor());
//            }

            std::cout << "irls trimmed "<< trimmed << " two euclidean iteration " << iter << "; inliers size " << num_inliers << "\n";
            if (iter != 0 && max_score < num_inliers) max_score = num_inliers;
            if (iter != 0 && min_score > num_inliers) min_score = num_inliers;
            avg_score += num_inliers;
        }
        std::cout << "max score " << max_score << "\n";
        std::cout << "min score " << min_score << "\n";
        std::cout << "avg score " << avg_score / num_lo_iters << "\n";
        std::cout << "-----------------------------------------------\n";

    }
    void runIRLSManhattanFourWeights (Model * model, Score * score, bool trimmed=false) {
        irls_model.setDescriptor(model->returnDescriptor());

        unsigned int max_score = 0;
        unsigned int min_score = 10000000;
        float avg_score = 0;
        unsigned int num_samples, num_inliers;
        for (unsigned int iter = 1; iter < num_lo_iters; iter++) {
            estimator->setModelParameters(irls_model.returnDescriptor());

            if (trimmed) {
                num_inliers = estimator->getInliersWeights(irls_model.threshold, inliers,
                                                           true, errors, false, nullptr, false, nullptr, true, weights_manh1, weights_manh2, weights_manh3, weights_manh4);
                num_samples = std::min(max_sample_size, num_inliers);
                if (num_samples <= model->sample_size) continue;
                std::sort (inliers, inliers+num_inliers, [&](int a, int b) {
                    return errors[a] < errors[b];
                });
                estimator->EstimateModelNonMinimalSample(inliers, num_samples, weights_manh1, weights_manh2, weights_manh3, weights_manh4, irls_model);
            } else {
                num_inliers = estimator->getInliersWeights(irls_model.threshold, inliers,
                                                           false, nullptr, false, nullptr, false, nullptr, true, weights_manh1, weights_manh2, weights_manh3, weights_manh4);
                num_samples = std::min(max_sample_size, num_inliers);
                if (num_samples <= model->sample_size) continue;
                uniformRandomGenerator.generateUniqueRandomSet(sample, num_samples, num_inliers - 1);
                for (unsigned int smpl = 0; smpl < num_samples; smpl++) {
                    sample[smpl] = inliers[sample[smpl]];
                }
                estimator->EstimateModelNonMinimalSample(sample, num_samples, weights_manh1, weights_manh2, weights_manh3, weights_manh4, irls_model);
            }
            irls_score.inlier_number = num_inliers;
            irls_score.score = irls_score.inlier_number;

//            if (irls_score.bigger(score)) {
//                score->copyFrom(irls_score);
//                model->setDescriptor(irls_model.returnDescriptor());
//            }

            std::cout << "irls trimmed " << trimmed << " four manhattan iteration " << iter << "; inliers size " << num_inliers << "\n";
            if (iter != 0 && max_score < num_inliers) max_score = num_inliers;
            if (iter != 0 && min_score > num_inliers) min_score = num_inliers;
            avg_score += num_inliers;
        }
        std::cout << "max score " << max_score << "\n";
        std::cout << "min score " << min_score << "\n";
        std::cout << "avg score " << avg_score / num_lo_iters << "\n";
        std::cout << "-----------------------------------------------\n";

    }
    void runIRLSIdentityWeights (Model * model, Score * score, bool trimmed=false) {
        irls_model.setDescriptor(model->returnDescriptor());

        unsigned int max_score = 0;
        unsigned int min_score = 10000000;
        float avg_score = 0;
        unsigned int num_samples, num_inliers;
        for (unsigned int iter = 1; iter < num_lo_iters; iter++) {
            estimator->setModelParameters(irls_model.returnDescriptor());

            if (trimmed) {
                num_inliers = estimator->getInliersWeights(irls_model.threshold, inliers,
                                                           true, errors, false, nullptr, false, nullptr, false, nullptr, nullptr, nullptr, nullptr);
                num_samples = std::min(max_sample_size, num_inliers);
                if (num_samples <= model->sample_size) continue;
                std::sort (inliers, inliers+num_inliers, [&](unsigned int a, unsigned int b) {
                    return errors[a] < errors[b];
                });
                estimator->EstimateModelNonMinimalSample(inliers, num_samples, irls_model);
            } else {
                num_inliers = estimator->getInliersWeights(irls_model.threshold, inliers,
                                                           false, nullptr, false, nullptr, false, nullptr, false, nullptr, nullptr, nullptr, nullptr);
                num_samples = std::min(max_sample_size, num_inliers);
//                if (num_samples <= model->sample_size) continue;
                uniformRandomGenerator.generateUniqueRandomSet(sample, num_samples, num_inliers - 1);
                for (unsigned int smpl = 0; smpl < num_samples; smpl++) {
                    sample[smpl] = inliers[sample[smpl]];
                }
                estimator->EstimateModelNonMinimalSample(sample, num_samples, irls_model);
            }
            irls_score.inlier_number = num_inliers;
            irls_score.score = irls_score.inlier_number;

//            if (irls_score.bigger(score)) {
//                score->copyFrom(irls_score);
//                model->setDescriptor(irls_model.returnDescriptor());
//            }

            std::cout << "irls trimmed " << trimmed << " identity weights iteration " << iter << "; inliers size " << num_inliers << "\n";
            if (iter != 0 && max_score < num_inliers) max_score = num_inliers;
            if (iter != 0 && min_score > num_inliers) min_score = num_inliers;
            avg_score += num_inliers;
        }
        std::cout << "max score " << max_score << "\n";
        std::cout << "min score " << min_score << "\n";
        std::cout << "avg score " << avg_score / num_lo_iters << "\n";
        std::cout << "-----------------------------------------------\n";

    }

    unsigned int getNumberIterations () override {
        return 0;
    }
};

#endif //USAC_IRLS_H
