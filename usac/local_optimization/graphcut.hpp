// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef USAC_GRAPHCUT_H
#define USAC_GRAPHCUT_H

#include "../precomp.hpp"

#include "../estimator/estimator.hpp"
#include "../../include/gco-v3.0/GCoptimization.h"
#include "local_optimization.hpp"
#include "../sampler/uniform_sampler.hpp"
#include "../random_generator/uniform_random_generator.hpp"


class GraphCut : public LocalOptimization {
protected:
    Estimator * estimator;
    Quality * quality;
    Score * gc_score;
    Model * gc_model;
    UniformRandomGenerator * uniform_random_generator;

    std::vector<std::vector<int>> neighbors_v;
    NeighborsSearch neighborsType;

    int knn, neighbor_number;
    unsigned int points_size, sample_limit, sample_size, lo_inner_iterations;
    float threshold, spatial_coherence, sqr_thr;

    int *inliers, *sample, *neighbors;
    float * errors;
public:
    unsigned int gc_iterations = 0;

    ~GraphCut() override {
        delete[] errors; delete[] inliers; delete[] sample;
        delete (gc_score); delete (gc_model); delete (uniform_random_generator);
    }
    
    GraphCut (Model * model, Estimator * estimator_, Quality * quality_, unsigned int points_size_) {
        spatial_coherence = model->spatial_coherence_gc;
        knn = model->k_nearest_neighbors;
        threshold = model->threshold;
        estimator = estimator_;
        quality = quality_;
        sqr_thr = 2 * threshold * threshold;
        points_size = points_size_;

        gc_score = new Score;
        gc_model = new Model (model);

        sample_limit = 7 * model->sample_size;

        sample_size = model->sample_size;

        errors = new float[points_size];
        inliers = new int [points_size];
        sample = new int [sample_limit];

        // set uniform random generator
        uniform_random_generator = new UniformRandomGenerator;
        uniform_random_generator->setSubsetSize(sample_limit);
        if (model->reset_random_generator) uniform_random_generator->resetTime();
        //

        lo_inner_iterations = model->lo_inner_iterations;

        // only for test
        gc_iterations = 0;
        //
    }

    void setNeighbors (cv::InputArray neighbors_, NeighborsSearch search) {
        assert(! neighbors_.empty());
        assert(search != NeighborsSearch::NullN);
        neighborsType = search;

        if (search == NeighborsSearch::Nanoflann) {
            neighbor_number = knn * points_size;
            neighbors = (int *) neighbors_.getMat().data;
        } else {
            neighbors_v = *(std::vector<std::vector<int>>*) neighbors_.getObj();
            neighbor_number = 0;
            for (int i = 0; i < points_size; i++) {
                neighbor_number += neighbors_v[i].size();
            }
        }
    }

    // calculate lambda
    void calculateSpatialCoherence (float inlier_number) {
        spatial_coherence = (points_size * (inlier_number / points_size)) / static_cast<float>(neighbor_number);
    }

	void labeling (const cv::Mat& model, Score * score, int * inliers);

    void GetModelScore (Model * best_model, Score * best_score) override {
        // improve best model by non minimal estimation
//        oneStepLO (best_model);

        bool is_best_model_updated = true;
        while (is_best_model_updated) {
            is_best_model_updated = false;

            // Build graph problem. Apply graph cut to G
            labeling(best_model->returnDescriptor(), gc_score, inliers);

            // if number of "virtual" inliers is too small then break
            if (gc_score->inlier_number <= sample_size) break;
            unsigned int labeling_inliers_size = gc_score->inlier_number;

            for (unsigned int iter = 0; iter < lo_inner_iterations; iter++) {
                // sample to generate min (|I_7m|, |I|)
                if (labeling_inliers_size > sample_limit) {
                    // generate random subset in range <0; |I|>
                    uniform_random_generator->generateUniqueRandomSet(sample, labeling_inliers_size-1);
                    // sample from inliers of labeling
                    for (unsigned int smpl = 0; smpl < sample_limit; smpl++) {
                        sample[smpl] = inliers[sample[smpl]];
                    }
                    if (! estimator->EstimateModelNonMinimalSample(sample, sample_limit, *gc_model)) {
                        break;
                    }
                } else {
                    if (iter > 0) {
                        /*
                         * If iterations are more than 0 and there are not enough inliers for random sampling,
                         * so break. Because EstimateModelNonMinimalSample for the same inliers gives same model, it
                         * is redundant to use it more than 1 time.
                         */
                        break;
                    }
                    if (! estimator->EstimateModelNonMinimalSample(inliers, labeling_inliers_size, *gc_model)) {
                        break;
                    }
                }

                quality->getNumberInliers(gc_score, gc_model->returnDescriptor());

                if (gc_score->bigger(best_score)) {
                    is_best_model_updated = true;
                    best_score->copyFrom(gc_score);
                    best_model->setDescriptor(gc_model->returnDescriptor());
                }

                // only for test
                gc_iterations++;
                //
            } // end of inner GC local optimization
        } // end of while loop
    }

private:
    void oneStepLO (Model * model) {
        /*
         * Do one step local optimization on min (|I|, max_I) before doing Graph Cut.
         * This LSQ must give better model for next GC labeling.
         */
        // use gc_score variable, but we are not getting gc score.
        quality->getNumberInliers(gc_score, model->returnDescriptor(), model->threshold, true, inliers);

        // return if not enough inliers
        if (gc_score->inlier_number <= model->sample_size)
            return;

        unsigned int one_step_lo_sample_limit = model->lo_sample_size;

        if (gc_score->inlier_number < one_step_lo_sample_limit) {
            // if score is less than limit number sample then take estimation of all inliers
            estimator->EstimateModelNonMinimalSample(inliers, gc_score->inlier_number, *model);
        } else {
            // otherwise take some inliers as sample at random
            if (model->sampler == SAMPLER::Prosac) {
                // if we use prosac sample, so points are ordered by some score,
                // so take first N inliers, because they have higher score
                for (unsigned int smpl = 0; smpl < one_step_lo_sample_limit; smpl++) {
                    sample[smpl] = inliers[smpl];
                }
            } else {
                uniform_random_generator->generateUniqueRandomSet(sample, one_step_lo_sample_limit, gc_score->inlier_number-1);
                for (unsigned int smpl = 0; smpl < one_step_lo_sample_limit; smpl++) {
                    sample[smpl] = inliers[sample[smpl]];
                }
            }

            estimator->EstimateModelNonMinimalSample(sample, one_step_lo_sample_limit, *model);
        }
    }
};

#endif //USAC_GRAPHCUT_H
