// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "ransac.hpp"
#include "../local_optimization/irls.hpp"
#include "../local_optimization/inner_local_optimization.hpp"
#include "../local_optimization/graphcut.hpp"
#include "../sprt.hpp"

#include "../sampler/prosac_sampler.hpp"
#include "../termination_criteria/prosac_termination_criteria.hpp"

void Ransac::run() {
    auto begin_time = std::chrono::steady_clock::now();

    std::vector<Model> models;
    models.emplace_back(Model(model));

    // Allocate max size of models for fundamental matrix
    // estimation to avoid reallocation
    if (model->estimator == ESTIMATOR::Fundamental ) {
        // for fundamental matrix can be up to 3 solutions
        models.emplace_back(Model(model));
        models.emplace_back(Model(model));
    } else if (model->estimator == ESTIMATOR::Essential) {
        // for essential matrix can be up to 10 solutions
        for (int sol = 0; sol < 9; sol++) {
            models.emplace_back(Model(model));
        }
    }

    Model best_model (model);
    unsigned int number_of_models;

    int sample [estimator->SampleNumber()];

    // prosac
    bool is_prosac = model->sampler == SAMPLER::Prosac;
    
    //------------- SPRT -------------------
    bool is_good_model, is_sprt = model->sprt;
    
    // LO
    bool LO = model->lo != LocOpt ::NullLO;
    bool GraphCutLO = model->lo == LocOpt ::GC;
    //------------------------------------------

    unsigned int iters = 0;
    unsigned int max_iters = model->max_iterations;

    while (iters < max_iters) {
        sampler->generateSample(sample);

        number_of_models = estimator->EstimateModel(sample, models);

        for (unsigned int i = 0; i < number_of_models; i++) {
            if (is_sprt) {
                is_good_model = sprt->verifyModelAndGetModelScore(&models[i], iters, best_score->inlier_number,
                                                                 current_score);
                if (!is_good_model) {
                    // do not skip bad model until predefined iterations reached
                    if (iters >= model->max_hypothesis_test_before_sprt) {
                        iters++;
                        continue;
                    }
                }
            } else {
                 quality->getScore(current_score, models[i].returnDescriptor());
            }

//            std::cout << current_score->inlier_number << "\n";
            if (current_score->bigger(best_score)) {

                // update current model and current score by inner and iterative local optimization
                // if inlier number is too small, do not update

                if (LO) {
                    local_optimization->GetModelScore (&models[i], current_score);
                }

                // copy current score to best score
                best_score->copyFrom(current_score);

                // remember best model
                best_model.setDescriptor (models[i].returnDescriptor());

                // Termination conditions:
                if (is_prosac) {
                    max_iters = ((ProsacTerminationCriteria *) termination_criteria)->
                            getUpBoundIterations(iters, best_model.returnDescriptor());
                } else {
                    max_iters = termination_criteria->getUpBoundIterations (best_score->inlier_number);
                }
                if (is_sprt) {
                    max_iters = std::min (max_iters, sprt->getUpperBoundIterations(best_score->inlier_number));
                }
            } // end of if so far the best score
        } // end loop of number of models
        iters++;
    } // end main while loop

    if (best_score->inlier_number == 0) {
        std::cout << "Best score is 0. Check it!\n";
        best_model.setDescriptor(cv::Mat_<float>::eye(3,3));
        exit (111);
    }

    // Graph Cut lo was set, but did not run, run it
    if (GraphCutLO && local_optimization->getNumberIterations() == 0) {
        // update best model and best score
        local_optimization->GetModelScore(&best_model, best_score);
    }

    Model non_minimal_model (model);

    unsigned int previous_non_minimal_num_inlier = 0;

    int * max_inliers = new int[points_size];
    // get inliers from the best model
    quality->getInliers(best_model.returnDescriptor(), max_inliers);

    for (unsigned int norm = 0; norm < 4 /* normalizations count */; norm++) {
        /*
         * TODO:
         * Calculate and Save Covariance Matrix and use it next normalization with adding or
         * extracting some points.
         */
        // estimate non minimal model with max inliers
        if (! estimator->EstimateModelNonMinimalSample(max_inliers, best_score->inlier_number, non_minimal_model)) {
            std::cout << "\033[1;31mNON minimal model completely failed!\033[0m \n";
            break;
        }

        quality->getScore(current_score, non_minimal_model.returnDescriptor(), model->threshold, true, max_inliers);

        // Priority is for non minimal model estimation
        // break if non minimal model score is less than 80% of the best minimal model score
        if ((float) current_score->inlier_number / best_score->inlier_number < 0.8) {
            break;
        }

        // if normalization score is less or equal, so next normalization is equal too, so break.
        if (current_score->inlier_number <= previous_non_minimal_num_inlier) {
            break;
        }

        previous_non_minimal_num_inlier = current_score->inlier_number;

        best_score->copyFrom(current_score);
        best_model.setDescriptor(non_minimal_model.returnDescriptor());
    }

    std::chrono::duration<float> fs = std::chrono::steady_clock::now() - begin_time;
    // ================= here is ending ransac main implementation ===========================

    // get final inliers from the best model
    quality->getInliers(best_model.returnDescriptor(), max_inliers);

    unsigned int num_lo_iters = model->lo == NullLO ? 0 : local_optimization->getNumberIterations();

    // Store results
    ransac_output = new RansacOutput (&best_model, max_inliers,
            std::chrono::duration_cast<std::chrono::microseconds>(fs).count(),
                                      best_score->inlier_number, iters, num_lo_iters);

    delete[] max_inliers;
}


void Ransac::run_debug() {
    auto begin_time = std::chrono::steady_clock::now();

    std::vector<Model> models;
    models.push_back (Model(model));

    if (model->estimator == ESTIMATOR::Fundamental ) {
        models.push_back (Model(model));
        models.push_back (Model(model));
    } else if (model->estimator == ESTIMATOR::Essential) {
        for (int sol = 0; sol < 9; sol++) {
            models.push_back(Model(model));
        }
    }

    Model * best_model = new Model (model);
    unsigned int number_of_models;

    int sample [estimator->SampleNumber()];

    bool is_prosac = model->sampler == SAMPLER::Prosac;

    bool is_good_model, is_sprt = model->sprt;

    bool LO = model->lo != LocOpt ::NullLO;
    bool GraphCutLO = model->lo == LocOpt ::GC;

    unsigned int iters = 0;
    unsigned int max_iters = model->max_iterations;

   long sampling_time = 0, min_estimation_time = 0, eval_time = 0, non_min_est_time = 0, loc_opt_time = 0;

    while (iters < max_iters) {
       // std::cout << "generate sample\n";
        auto t = std::chrono::steady_clock::now();
        sampler->generateSample(sample);
        sampling_time+=std::chrono::duration_cast<std::chrono::microseconds>
               (std::chrono::steady_clock::now() - t).count();

        // std::cout << "samples are generated\n";
        auto t2 = std::chrono::steady_clock::now();
        number_of_models = estimator->EstimateModel(sample, models);
        min_estimation_time += std::chrono::duration_cast<std::chrono::microseconds>
               (std::chrono::steady_clock::now() - t2).count();

        // std::cout << "minimal model estimated\n";

        for (unsigned int i = 0; i < number_of_models; i++) {
//             std::cout << i << "-th model\n";

            if (is_sprt) {
//                std::cout << "sprt verify\n";
                is_good_model = sprt->verifyModelAndGetModelScore(&models[i], iters, best_score->inlier_number,
                                                                 current_score);
//                std::cout << "sprt verified\n";

                if (!is_good_model) {
//                    std::cout << "model is bad\n";

                    // do not skip bad model until predefined iterations reached
                    if (iters >= model->max_hypothesis_test_before_sprt) {
                        iters++;
//                        std::cout << "skip bad model in iteration " << iters << "\n";
                        continue;
                    }
//                    else {
//                        std::cout << "sprt " << current_score->inlier_number << "\n";
//                        quality->getNumberInliers(current_score, models[i]);
//                        std::cout << "std " << current_score->inlier_number << "\n";
//                    }
                }
//                else {
//                    std::cout << "model is good\n";
//                }
            } else {
//                std::cout << "Get quality score\n";

               auto t = std::chrono::steady_clock::now();
                 quality->getScore(current_score, models[i].returnDescriptor());
               eval_time += std::chrono::duration_cast<std::chrono::microseconds>
                       (std::chrono::steady_clock::now() - t).count();
            }
//
          // std::cout << "Ransac, iteration " << iters << "; score " << current_score->inlier_number << "\n";
//            std::cout << models[i]->returnDescriptor() << "\n\n";

            if (current_score->bigger(best_score)) {

//                  std::cout << "update best score\n";

                // update current model and current score by inner and iterative local optimization
                // if inlier number is too small, do not update

                auto t3 = std::chrono::steady_clock::now();
                if (LO) {
                    local_optimization->GetModelScore (&models[i], current_score);
                }
                loc_opt_time += std::chrono::duration_cast<std::chrono::microseconds>
                        (std::chrono::steady_clock::now() - t3).count();

                // copy current score to best score
                best_score->copyFrom(current_score);

                // remember best model
                best_model->setDescriptor (models[i].returnDescriptor());

//                  std::cout << "Ransac, update best score " << best_score->inlier_number << '\n';

                // Termination conditions:
                if (is_prosac) {
                    max_iters = ((ProsacTerminationCriteria *) termination_criteria)->
                            getUpBoundIterations(iters, best_model->returnDescriptor());
                } else {
                    max_iters = termination_criteria->getUpBoundIterations (best_score->inlier_number);
                }
                if (is_sprt) {
                    max_iters = std::min (max_iters, sprt->getUpperBoundIterations(best_score->inlier_number));
                }
//                 std::cout << "max iters prediction = " << max_iters << '\n';
            } // end of if so far the best score
        } // end loop of number of models
        iters++;
    } // end main while loop

//    std::cout << "end:\n";

    if (best_score->inlier_number == 0) {
        std::cout << "Best score is 0. Check it!\n";
        best_model->setDescriptor(cv::Mat_<float>::eye(3,3));
        exit (111);
    }

    if (GraphCutLO && local_optimization->getNumberIterations() == 0) {
        local_optimization->GetModelScore(best_model, best_score);
    }

//    std::cout << "Calculate Non minimal model\n";

    Model *non_minimal_model = new Model (model);

   // std::cout << "end best inl num " << best_score->inlier_number << '\n';

    unsigned int previous_non_minimal_num_inlier = 0;

    int * max_inliers = new int[points_size];
    // get inliers from the best model
    quality->getInliers(best_model->returnDescriptor(), max_inliers);

   auto t = std::chrono::steady_clock::now();
    for (unsigned int norm = 0; norm < 4 /* normalizations count */; norm++) {
//        std::cout << "estimate non minimal\n";
//        std::cout << best_score->inlier_number << " -\n";
        // estimate non minimal model with max inliers
        if (! estimator->EstimateModelNonMinimalSample(max_inliers, best_score->inlier_number, *non_minimal_model)) {
            std::cout << "\033[1;31mNON minimal model completely failed!\033[0m \n";
            break;
        }

        //
//        std::cout << "get non minimal score\n";max_inliers
        quality->getScore(current_score, non_minimal_model->returnDescriptor(), model->threshold, true, max_inliers);
//        std::cout << "end get non minimal score\n";

        // Priority is for non minimal model estimation
       // std::cout << "non minimal score " << current_score->inlier_number << '\n';

        // break if non minimal model score is less than 80% of the best minimal model score
        if ((float) current_score->inlier_number / best_score->inlier_number < 0.8) {
//            std::cout << "break; non minimal score is significanlty worse than best score: RSC* "
//                         << best_score->inlier_number << " vs PCA " << current_score->inlier_number <<"\n";
            break;
        }

        // if normalization score is less or equal, so next normalization is equal too, so break.
        if (current_score->inlier_number <= previous_non_minimal_num_inlier) {
            // std::cout << "break; previous non minimal score is the same.\n";
            break;
        }

        previous_non_minimal_num_inlier = current_score->inlier_number;

        best_score->copyFrom(current_score);
        best_model->setDescriptor(non_minimal_model->returnDescriptor());
    }
   non_min_est_time = std::chrono::duration_cast<std::chrono::microseconds>
           (std::chrono::steady_clock::now() - t).count();

   std::cout << "sampling time " << sampling_time << "\n";
   std::cout << "evaluation time " << eval_time << "\n";
   std::cout << "minimal estimation time " << min_estimation_time << "\n";
   std::cout << "non minimal estimation time " << non_min_est_time << "\n";
   std::cout << "local optimization time " << loc_opt_time << "\n";

    std::chrono::duration<float> fs = std::chrono::steady_clock::now() - begin_time;
    // ================= here is ending ransac main implementation ===========================
//    std::cout << "get results\n";

    // get final inliers from the best model
    quality->getInliers(best_model->returnDescriptor(), max_inliers);

    unsigned int num_lo_iters = model->lo == NullLO ? 0 : local_optimization->getNumberIterations();

    // Store results
    ransac_output = new RansacOutput (best_model, max_inliers,
            std::chrono::duration_cast<std::chrono::microseconds>(fs).count(),
                                      best_score->inlier_number, iters, num_lo_iters);

    delete[] max_inliers;
    delete (best_model);
}
