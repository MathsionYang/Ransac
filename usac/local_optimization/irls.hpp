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

    float *errors, *weights1, *weights2, *weights3, *weights4;
    int *inliers, *sample;
    unsigned int points_size, max_sample_size, num_lo_iters;
    float threshold;
    enum WEIGHTS {DEFAULT, EUCLIDEAN_1, EUCLIDEAN_2, MANHATTAN_4, SAMPSON};

public:
    ~Irls() override {
        delete[] errors;
        delete[] weights1; delete[] weights2; delete[] weights3; delete[] weights4;
        delete[] inliers; delete[] sample;
    }

    Irls (Model * model, Estimator * estimator_, Quality * quality_, unsigned int points_size_) : irls_model (model){
        max_sample_size = model->lo_sample_size;
        points_size = points_size_;

        errors = new float[points_size];
        weights1 = new float [points_size];
        weights2 = new float [points_size];
        weights3 = new float [points_size];
        weights4 = new float [points_size];

        inliers = new int[points_size_];
        sample = new int [max_sample_size];

        quality = quality_;
        estimator = estimator_;
        threshold = model->threshold;
        num_lo_iters = 10; // model->lo_inner_iterations;

        if (model->reset_random_generator) {
            uniformRandomGenerator.resetTime();
        }
        uniformRandomGenerator.setSubsetSize(max_sample_size);
    }

    void GetModelScore (Model * model, Score * score) override {
//            std::sort(errors.begin(), errors.end());
//            std::vector<float> y(errors.size());
//            std::iota(y.begin(), y.end(), 1);
//            matplotlibcpp::plot(y, errors, "r-");
//            matplotlibcpp::show();
//            matplotlibcpp::save("../results/1.png");
//            exit(0);

        Model model1 (model), model2 (model), model3 (model), model4 (model), model5 (model);

        RansacScore score1, score2, score3, score4, score5;
        score1.copyFrom(score);
        score2.copyFrom(score);
        score3.copyFrom(score);
        score4.copyFrom(score);
        score5.copyFrom(score);


        runIRLS (&model1, &score1, WEIGHTS::EUCLIDEAN_1);
        runIRLS (&model2, &score2, WEIGHTS::EUCLIDEAN_2);
        runIRLS (&model3, &score3, WEIGHTS::MANHATTAN_4);
        runIRLS (&model3, &score3, WEIGHTS::SAMPSON);
        runIRLS (&model4, &score4, WEIGHTS::DEFAULT);

        exit (0);
        runTrimmedIRLS (&model5, &score5, WEIGHTS::DEFAULT);
    }

    /*
     * Dilemma:
     * Update weights after new maximum score reached.
     * Disadvantage: if the best model is bad, so sampling from
     * inliers of this model does not significantly improve the best model.
     * Advantage: faster
     *
     * Or update weights in every iteration.
     * Advantage: if model is bad, after reweighting in every iteration
     * change model to good.
     * Disadvantage: slower.
     *
     * Probably it is 2 different approaches, which both we must try.
     */
    void runIRLS (Model * best_model, Score * best_score, WEIGHTS weights) {
        unsigned int max_score = 0;
        unsigned int min_score = 10000000;
        float avg_score = 0;

        unsigned int num_inliers = updateWeights(weights, best_model->returnDescriptor(), best_model->threshold);

        std::cout << "BEGIN SCORE " << num_inliers << "\n";

        if (num_inliers <= best_model->sample_size) {
            std::cout << "not enough inliers for IRLS!\n";
            return;
        }

        for (unsigned int iter = 0; iter < num_lo_iters; iter++) {
            if (num_inliers <= best_model->sample_size) break;

            if (num_inliers > max_sample_size) {
                uniformRandomGenerator.generateUniqueRandomSet(sample, num_inliers - 1);
                for (unsigned int smpl = 0; smpl < max_sample_size; smpl++) {
                    sample[smpl] = inliers[sample[smpl]];
                }
                if (weights == WEIGHTS::DEFAULT) {
                    estimator->EstimateModelNonMinimalSample(sample, max_sample_size, irls_model);
                } else if (weights == WEIGHTS::EUCLIDEAN_1) {
                    estimator->EstimateModelNonMinimalSample(sample, max_sample_size, weights1, irls_model);
                } else if (weights == WEIGHTS::EUCLIDEAN_2) {
                    estimator->EstimateModelNonMinimalSample(sample, max_sample_size, weights1, weights2, irls_model);
                } else if (weights == WEIGHTS::MANHATTAN_4) {
                    estimator->EstimateModelNonMinimalSample(sample, max_sample_size, weights1, weights2, weights3, weights4, irls_model);
                } else if (weights == WEIGHTS::SAMPSON) {
                    estimator->EstimateModelNonMinimalSample(sample, max_sample_size, weights1, irls_model);
                } else {
                    std::cout << "UNDEFINED WEIGHTS!\n";
                    break;
                }
            } else {
//                if (iter > 0) break;
                if (weights == WEIGHTS::DEFAULT) {
                    estimator->EstimateModelNonMinimalSample(inliers, num_inliers, irls_model);
                } else if (weights == WEIGHTS::EUCLIDEAN_1) {
                    estimator->EstimateModelNonMinimalSample(inliers, num_inliers, weights1, irls_model);
                } else if (weights == WEIGHTS::EUCLIDEAN_2) {
                    estimator->EstimateModelNonMinimalSample(inliers, num_inliers, weights1, weights2, irls_model);
                } else if (weights == WEIGHTS::MANHATTAN_4) {
                    estimator->EstimateModelNonMinimalSample(inliers, num_inliers, weights1, weights2, weights3, weights4, irls_model);
                } else if (weights == WEIGHTS::SAMPSON) {
                    estimator->EstimateModelNonMinimalSample(inliers, num_inliers, weights1, irls_model);
                } else {
                    std::cout << "UNDEFINED WEIGHTS!\n";
                    break;
                }
            }

//            quality->getScore(&irls_score, irls_model.returnDescriptor());
            num_inliers = updateWeights(weights, irls_model.returnDescriptor(), irls_model.threshold);
            irls_score.inlier_number = num_inliers;
            irls_score.score = num_inliers;

            std::cout << "irls, |weights = " << weights << "| iteration " << iter << "; inliers size " << irls_score.inlier_number << "\n";

//            if (irls_score.bigger(best_score)) {
//                best_score->copyFrom(irls_score);
//                best_model->setDescriptor(irls_model.returnDescriptor());
//                num_inliers = updateWeights(weights, best_model->returnDescriptor(), best_model->threshold);
//            }

            if (max_score < irls_score.inlier_number) max_score = irls_score.inlier_number;
            if (min_score > irls_score.inlier_number) min_score = irls_score.inlier_number;
            avg_score += irls_score.inlier_number;
        }
        std::cout << "max score " << max_score << "\n";
        std::cout << "min score " << min_score << "\n";
        std::cout << "avg score " << avg_score / num_lo_iters << "\n";
        std::cout << "-----------------------------------------------\n";
    }


    unsigned int updateWeights (WEIGHTS weights, const cv::Mat &best_model, float threshold, bool trimmed=false) {
        estimator->setModelParameters(best_model);
        unsigned int num_inliers = 0;

        // Note you can use only one type of weights, do not combine them!
        num_inliers = estimator->getInliersWeights(threshold, inliers,
                                                   trimmed, errors,
                                                   weights == WEIGHTS::EUCLIDEAN_1, weights1,
                                                   weights == WEIGHTS::EUCLIDEAN_2, weights2,
                                                   weights == WEIGHTS::SAMPSON, weights1,
                                                   weights == WEIGHTS::MANHATTAN_4, weights1, weights2, weights3, weights4);

        return num_inliers;
    }

    void runTrimmedIRLS (Model * best_model, Score * best_score, WEIGHTS weights) {
        unsigned int max_score = 0;
        unsigned int min_score = 10000000;
        float avg_score = 0;

        unsigned int num_inliers = updateWeights(weights, best_model->returnDescriptor(), best_model->threshold, true);

        std::cout << "BEGIN SCORE " << num_inliers << "\n";

        if (num_inliers <= best_model->sample_size) {
            std::cout << "not enough inliers for IRLS!\n";
            return;
        }

        bool is_model_updated = true;

        for (unsigned int iter = 0; iter < num_lo_iters; iter++) {
            if (! is_model_updated) break;

            if (num_inliers > max_sample_size) {
                std::sort (inliers, inliers+max_sample_size, [&](int a, int b) {
                    return errors[inliers[a]] < errors[inliers[b]];
                });
                if (weights == WEIGHTS::DEFAULT) {
                    estimator->EstimateModelNonMinimalSample(inliers, max_sample_size, irls_model);
                } else if (weights == WEIGHTS::EUCLIDEAN_1) {
                    estimator->EstimateModelNonMinimalSample(inliers, max_sample_size, weights1, irls_model);
                } else if (weights == WEIGHTS::EUCLIDEAN_2) {
                    estimator->EstimateModelNonMinimalSample(inliers, max_sample_size, weights1, weights2, irls_model);
                } else if (weights == WEIGHTS::MANHATTAN_4) {
                    estimator->EstimateModelNonMinimalSample(inliers, max_sample_size, weights1, weights2, weights3, weights4, irls_model);
                } else if (weights == WEIGHTS::SAMPSON) {
                    estimator->EstimateModelNonMinimalSample(inliers, max_sample_size, weights1, irls_model);
                } else {
                    std::cout << "UNDEFINED WEIGHTS!\n";
                    break;
                }
            } else {
                if (iter > 0) break;
                if (weights == WEIGHTS::DEFAULT) {
                    estimator->EstimateModelNonMinimalSample(inliers, num_inliers, irls_model);
                } else if (weights == WEIGHTS::EUCLIDEAN_1) {
                    estimator->EstimateModelNonMinimalSample(inliers, num_inliers, weights1, irls_model);
                } else if (weights == WEIGHTS::EUCLIDEAN_2) {
                    estimator->EstimateModelNonMinimalSample(inliers, num_inliers, weights1, weights2, irls_model);
                } else if (weights == WEIGHTS::MANHATTAN_4) {
                    estimator->EstimateModelNonMinimalSample(inliers, num_inliers, weights1, weights2, weights3, weights4, irls_model);
                } else if (weights == WEIGHTS::SAMPSON) {
                    estimator->EstimateModelNonMinimalSample(inliers, num_inliers, weights1, irls_model);
                } else {
                    std::cout << "UNDEFINED WEIGHTS!\n";
                    break;
                }
            }

            quality->getScore(&irls_score, irls_model.returnDescriptor());

            std::cout << "trimmed irls, |weights = " << weights << "| iteration " << iter << "; inliers size " << irls_score.inlier_number << "\n";

            if (irls_score.better(best_score)) {
                best_score->copyFrom(irls_score);
                best_model->setDescriptor(irls_model.returnDescriptor());
                num_inliers = updateWeights(weights, best_model->returnDescriptor(), best_model->threshold, true);
                is_model_updated = true;
            } else {
                is_model_updated = false;
            }

            if (max_score < irls_score.inlier_number) max_score = irls_score.inlier_number;
            if (min_score > irls_score.inlier_number) min_score = irls_score.inlier_number;
            avg_score += irls_score.inlier_number;
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
