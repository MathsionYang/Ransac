#ifndef USAC_RANSACLOCALOPTIMIZATION_H
#define USAC_RANSACLOCALOPTIMIZATION_H

#include "local_optimization.hpp"
#include "../sampler/uniform_sampler.hpp"
#include "../random_generator/uniform_random_generator.hpp"
#include "iterative_local_optimization.hpp"
#include "../quality/ransac_quality.hpp"

/*
 * Reference:
 * http://cmp.felk.cvut.cz/~matas/papers/chum-dagm03.pdf
 */

class InnerLocalOptimization : public LocalOptimization {
private:
    Quality * quality;
    Estimator * estimator;
    RansacScore lo_score;
    Model lo_model;
    UniformRandomGenerator uniform_random_generator;
    IterativeLocalOptimization * iterativeLocalOptimization;

    int *inliers_of_best_model, *lo_sample;
    unsigned int lo_inner_max_iterations, sample_limit;
    bool run_iterative;
    unsigned int lo_inner_iters;
public:

    ~InnerLocalOptimization () override {
        delete[] lo_sample; delete[] inliers_of_best_model;
        delete (iterativeLocalOptimization);
    }

    InnerLocalOptimization (Model *model, Estimator *estimator_, Quality *quality_, unsigned int points_size)
    : lo_model (model) {

        estimator = estimator_;
        quality = quality_;

        lo_inner_max_iterations = model->lo_inner_iterations;
        sample_limit = model->lo_sample_size;

        // Allocate max memory to avoid reallocation
        inliers_of_best_model = new int [points_size];
        lo_sample = new int [sample_limit];

        // ------------- set random generator --------------
        uniform_random_generator.setSubsetSize(sample_limit);
        if (model->reset_random_generator) uniform_random_generator.resetTime();
        // -----------------------------------

        // iterative lo
        run_iterative = model->lo == LocOpt ::InItFLORsc || LocOpt ::InItLORsc;
        if (run_iterative) {
            iterativeLocalOptimization = new IterativeLocalOptimization(model, estimator, quality, points_size);
        }
        // ----------------------

        lo_inner_iters = 0;
    }

    /*
     * Implementation of Locally Optimized Ransac
     * Inner + Iterative
     */
    void GetModelScore (Model * best_model, Score * best_score) override {
        // return if there are not many inliers for LO.
        if (best_score->inlier_number < best_model->lo_sample_size) return;

        // get inliers from so far the best model.
        quality->getInliers(best_model->returnDescriptor(), inliers_of_best_model);

        // Inner Local Optimization Ransac.
        for (unsigned int iters = 0; iters < lo_inner_max_iterations; iters++) {
            // Generate sample of lo_sample_size from inliers from the best model.
            if (best_score->inlier_number > sample_limit) {
                // if there are many inliers take limited number at random.
                uniform_random_generator.generateUniqueRandomSet(lo_sample, best_score->inlier_number-1);
                // get inliers from maximum inliers from lo
                for (unsigned int smpl = 0; smpl < sample_limit; smpl++) {
                    lo_sample[smpl] = inliers_of_best_model[lo_sample[smpl]];
                }
            
                if (!estimator->LeastSquaresFitting(lo_sample, sample_limit, lo_model)) continue;
            } else {
                // if model was not updated in first iteration, so break.
                if (iters > 0) break;
                // if inliers are less than limited number of sample then take all of them for estimation
                // if it fails -> end Lo.
                if (!estimator->LeastSquaresFitting(inliers_of_best_model, best_score->inlier_number, lo_model)) return;
            }

            if (run_iterative) {
                iterativeLocalOptimization->GetModelScore(&lo_model, &lo_score);
            } else {
                // just get score of non minimal inner ransac model
                quality->getScore(&lo_score, lo_model.returnDescriptor(), lo_model.threshold);
            }
            
            if (lo_score.better(best_score)) {
                // update best model
                best_model->setDescriptor(lo_model.returnDescriptor());
                best_score->copyFrom(lo_score);
                // update also inliers of the best model.
                quality->getInliers(best_model->returnDescriptor(), inliers_of_best_model);
            }

            // only for test
            lo_inner_iters++;
            //
        }
    }
    unsigned int getNumberIterations () override {
        unsigned int lo_iterative_iters = 0;
        if (run_iterative) {
            lo_iterative_iters = iterativeLocalOptimization->getNumberIterations();
        }
        return lo_inner_iters + lo_iterative_iters;
    }
};

#endif //USAC_RANSACLOCALOPTIMIZATION_H
