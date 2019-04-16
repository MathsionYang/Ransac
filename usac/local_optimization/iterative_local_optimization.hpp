// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef USAC_ITERATIVELOCALOPTIMIZATION_H
#define USAC_ITERATIVELOCALOPTIMIZATION_H

#include "../quality/quality.hpp"
#include "../random_generator/uniform_random_generator.hpp"
#include "../quality/ransac_quality.hpp"

class IterativeLocalOptimization : public LocalOptimization{
private:
    unsigned int max_iters, threshold_multiplier, sample_limit;
    bool is_sample_limit;
    float threshold_step, threshold, new_threshold;
    UniformRandomGenerator uniformRandomGenerator;
    Estimator * estimator;
    Quality * quality;
    RansacScore lo_score;
    Model lo_model;
    int *lo_sample, *max_virtual_inliers;
    unsigned int lo_iterative_iters;
public:

    ~IterativeLocalOptimization() override {
        delete[] max_virtual_inliers;
        if (is_sample_limit) {
            delete[] lo_sample;
        }
    }
    IterativeLocalOptimization (Model * model, Estimator * estimator_, Quality * quality_, unsigned int points_size) :lo_model(model) {
        max_iters = model->lo_iterative_iterations;
        threshold_multiplier = model->lo_threshold_multiplier;
        is_sample_limit = model->lo == LocOpt ::InItFLORsc || model->lo == LocOpt ::ItFLORsc;
        sample_limit = model->lo_sample_size;
        if (is_sample_limit) {
            lo_sample = new int [sample_limit];
        }
        max_virtual_inliers = new int[points_size];

        threshold = model->threshold;
        new_threshold = model->lo_threshold_multiplier * model->threshold;
        /*
         * reduce multiplier threshold K·θ by this number in each iteration.
         * In the last iteration there be original threshold θ.
         */
        threshold_step = (new_threshold - threshold) / max_iters;

        // ------------- set random generator --------------
        uniformRandomGenerator.setSubsetSize(sample_limit);
        if (model->reset_random_generator) uniformRandomGenerator.resetTime();
        // -----------------------------------

        estimator = estimator_;
        quality = quality_;

        lo_iterative_iters = 0;
    }

    /*
     * Iterative LO Ransac
     * Get inliers of the best model or the model of Inner LO Ransac
     * Reduce threshold of current model
     * Estimate model parametres from limited sample size of lo model.
     * Evaluate model
     * Get inliers
     * Repeat until iteration < lo iterative iterations
     */

    void GetModelScore (Model * best_model, Score * best_score) override {
        // Start evaluating a model with new threshold.
        // multiply threshold K * θ
        lo_model.threshold = new_threshold;
        // get max virtual inliers. Note that they are nor real inliers, because we got them with bigger threshold.
        quality->getScore(&lo_score, best_model->returnDescriptor(), lo_model.threshold, true, max_virtual_inliers);
        
        unsigned int max_virtual_inliers_number = lo_score.inlier_number;
        
        // return if there is small number of inliers
        if (lo_score.inlier_number < sample_limit) return;

        for (unsigned int iterations = 0; iterations < max_iters; iterations++) {
            lo_model.threshold -= threshold_step;

            if (sample_limit) {
                // if there are more inliers than limit for sample size then generate at random
                // sample from LO model.
                uniformRandomGenerator.generateUniqueRandomSet(lo_sample, lo_score.inlier_number-1);
                for (unsigned int smpl = 0; smpl < sample_limit; smpl++) {
                    lo_sample[smpl] = max_virtual_inliers[lo_sample[smpl]];
                }
                if (! estimator->LeastSquaresFitting(lo_sample, sample_limit, lo_model)) continue;
            
            } else {
                // break if failed, very low probability that it will not fail in next iterations
                if (!estimator->LeastSquaresFitting(max_virtual_inliers, lo_score.inlier_number, lo_model)) break;
            }

            quality->getScore(&lo_score, lo_model.returnDescriptor(), lo_model.threshold);

            // only for test
            lo_iterative_iters++;
            //

            // In case of unlimited sample:
            // break if the best score is bigger, because after decreasing
            // threshold lo score could not be bigger in next iterations.
            if (! sample_limit && best_score->better(lo_score)) break;
            
            // Update max virtual inliers
            if (lo_score.inlier_number > max_virtual_inliers_number) {
                max_virtual_inliers_number = lo_score.inlier_number;
                quality->getInliers(lo_model.returnDescriptor(), max_virtual_inliers, lo_model.threshold);
            }
        }

        if (fabsf (lo_model.threshold - threshold) < 0.0001) {
            std::cout << "Success\n";
            // Success, threshold does not differ
            // last score correspond to user-defined threshold. Inliers are real.
            if (lo_score.better(best_score)) {
                // update best model and best score
                best_score->copyFrom(lo_score);
                best_model->setDescriptor(lo_model.returnDescriptor());
            }
        } else {
            std::cout << "Fail\n";
        }

    }

    unsigned int getNumberIterations () override {
        return lo_iterative_iters;
    }
};

#endif //USAC_ITERATIVELOCALOPTIMIZATION_H
