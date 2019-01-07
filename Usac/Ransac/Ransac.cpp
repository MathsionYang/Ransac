#include "Ransac.h"
#include "../LocalOptimization/RansacLocalOptimization.h"
#include "../Estimator/DLT/DLT.h"
#include "../LocalOptimization/GraphCut.h"
#include "../SPRT.h"

#include "../Sampler/ProsacSampler.h"
#include "../TerminationCriteria/ProsacTerminationCriteria.h"
#include "../Estimator/HomographyEstimator.h"
#include "../LocalOptimization/GreedyLocalOptimization.h"
#include "../LocalOptimization/IRLS.h"
#include "../LocalOptimization/SortedLO.h"

int getPointsSize (cv::InputArray points) {
//    std::cout << points.getMat(0).total() << '\n';

    if (points.isVector()) {
        return points.size().width;
    } else {
        return points.getMat().rows;
    }
}

void Ransac::run(cv::InputArray input_points) {
    auto begin_time = std::chrono::steady_clock::now();

    // todo: initialize (= new) estimator, quality, sampler and others here...

    /*
     * Check if all components are initialized and safe to run
     * todo: add more criteria
     */
    assert(model->estimator != ESTIMATOR::NullE && model->sampler != SAMPLER::NullS);
    assert(!input_points.empty());
    assert(estimator != nullptr);
    assert(model != nullptr);
    assert(quality != nullptr);
    assert(sampler != nullptr);
    assert(termination_criteria != nullptr);
    assert(sampler->isInit());
    assert(model->neighborsType != NullN);

    int points_size = getPointsSize(input_points);

//   std::cout << "Points size " << points_size << '\n';

    // initialize termination criteria
    termination_criteria->init(model, points_size);

    // initialize quality
    quality->init(points_size, model->threshold, estimator);


    Score *best_score = new Score, *current_score = new Score;

    std::vector<Model*> models;
    models.push_back (new Model(model));

    // Allocate max size of models for fundamental matrix
    // estimation to avoid reallocation
    if (model->estimator == ESTIMATOR::Fundamental) {
        models.push_back (new Model(model));
        models.push_back (new Model(model));
    }
    Model * best_model = new Model (model);
    unsigned int number_of_models;


    //--------------- Prosac ---------------------
    ProsacTerminationCriteria * prosac_termination_criteria;
    ProsacSampler * prosac_sampler;
    bool is_prosac = model->sampler == SAMPLER::Prosac;
    if (is_prosac) {
        prosac_sampler = (ProsacSampler *) sampler;
        prosac_termination_criteria = (ProsacTerminationCriteria *) termination_criteria;
    }
    //--------------------------------------------

    /*
     * Allocate inliers of points_size, to avoid reallocation in getModelScore()
     */
    int * inliers = new int[points_size];
    int * sample = new int[estimator->SampleNumber()];

    // ------------- Standard Local Optimization
    bool LO = model->LO;
    LocalOptimization * lo_ransac;
    if (LO) {
        lo_ransac = new RansacLocalOptimization (model, sampler, termination_criteria, quality, estimator, points_size);
    }
    //--------------------------------------

    //------------- SPRT -------------------
    bool SprtLO = model->Sprt;
    SPRT * sprt;
    bool is_good_model;
    if (SprtLO) {
        sprt = new SPRT;
        sprt->initialize(estimator, model, points_size, model->reset_random_generator);
    }
    //--------------------------------------------

    //---------- Graph cut local optimization ----------
    bool GraphCutLO = model->GraphCutLO;
    GraphCut * graphCut;
    if (GraphCutLO) {
        graphCut = new GraphCut;
        graphCut->init(points_size, model, estimator, quality, model->neighborsType);
        if (model->neighborsType == NeighborsSearch::Nanoflann) {
            graphCut->setNeighbors(neighbors);
        } else {
            graphCut->setNeighbors(neighbors_v);
        }
    }
    unsigned int gc_runs = 0;
    //------------------------------------------

    // if we have small number of data points, so LO will run with quarter of them,
    // otherwise it requires at least 3 sample size.
    int min_inlier_count_for_LO = std::min (points_size/4,  3 * (int)model->sample_size);

    int iters = 0;
    int max_iters = model->max_iterations;

    // delete, just for test
//    int * best_sample = new int[4];

    // ------------ Iterated Reweighted Least Squares ------------------
    IRLS * irls = new IRLS(points_size, model, estimator, quality);
    // ------------------------------------------------------

    int * inliers_hist = (int *) calloc (points_size, sizeof(int));

    while (iters < max_iters) {

        if (is_prosac) {
            ((ProsacSampler *)sampler)->generateSampleProsac (sample, prosac_termination_criteria->getStoppingLength());
        } else {
            sampler->generateSample(sample);
        }

//      debug
//        bool eq = false;
//        for (int s = 0; s < model->sample_size; s++) {
//            std::cout << sample[s] << " ";
//            for (int j = 0; j < model->sample_size; j++) {
//                if (s == j) continue;
//                if (sample[s] == sample[j]) {
//                    eq = true;
//                }
//            }
//        }
//        std::cout << "\n";
//        if (eq) std::cout << "SAMPLE EQUAL\n";

//         std::cout << "samples are generated\n";

        number_of_models = estimator->EstimateModel(sample, models);

//         std::cout << "minimal model estimated\n";

        for (int i = 0; i < number_of_models; i++) {
//             std::cout << i << "-th model\n";

            if (SprtLO) {
//                std::cout << "sprt verify\n";
                is_good_model = sprt->verifyModelAndGetModelScore(models[i], iters,
                        std::max (best_score->inlier_number, current_score->inlier_number), current_score);
//                std::cout << "sprt verified\n";

                if (!is_good_model) {
                    iters++;
//                    std::cout << "model is bad\n";

                    // do not skip bad model until predefined iterations reached
                    if (iters >= model->max_hypothesis_test_before_sprt) {
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
                // quality->getNumberInliers(current_score, models[i]);
                quality->getNumberInliers(current_score, models[i], true, inliers);   
            }

            for (int p = 0; p < current_score->inlier_number; p++) {
                inliers_hist[inliers[p]]++;
            }

            // 100 is very small history. 300 is even better
            // Makes sense to do it only once or every e.g. 200 iterations
            // Almost always better than random non minimal sampling from inliers 
            // of the best model. But there are some (a little) cases, where mle-histogram estimation
            // is worse (possible improvements?).
            if (iters == 200) {
                int * arr = new int[points_size];
                for (int p = 0; p < points_size; p++) {
                    arr[p] = p;
                }
                std::sort (arr, arr+points_size, [&](int a, int b) {
                    return inliers_hist[a] > inliers_hist[b];
                });
                
                int max_sample_gen = 14;
                int * sample_ = new int[max_sample_gen];
                for (int s = 0; s < max_sample_gen; s++) {
                    sample_[s] = arr[s];
                    std::cout << inliers_hist[sample_[s]] << " ";
                }
                std::cout << "\n";
                Model * mlemodel = new Model(model);
                Score * mlescore = new Score;
                estimator->EstimateModelNonMinimalSample(sample_, max_sample_gen, *mlemodel);
                quality->getNumberInliers(mlescore, mlemodel);
                std::cout << "best score " << best_score->inlier_number << "\n";
                std::cout << "histogram mle score " << mlescore->inlier_number << "\n";
                if (mlescore->bigger(current_score)) {
                    current_score->copyFrom(mlescore);
                    models[i]->setDescriptor(mlemodel->returnDescriptor());
                }

                // debug 
                // random non minimal sampling
                Score * r_score = new Score;
                Model * r_model = new Model(model);
                quality->getNumberInliers(r_score, best_model, true, inliers);
                UniformRandomGenerator * uniformRandomGenerator = new UniformRandomGenerator;
                uniformRandomGenerator->resetTime();
                uniformRandomGenerator->setSubsetSize(max_sample_gen);
                uniformRandomGenerator->resetGenerator(0, r_score->inlier_number-1);
                uniformRandomGenerator->generateUniqueRandomSet(sample_);
                for (int i = 0; i < max_sample_gen; i++) {
                    sample_[i] = inliers[sample_[i]];
                }
                estimator->EstimateModelNonMinimalSample(sample_, max_sample_gen, *r_model);
                quality->getNumberInliers(r_score, r_model);
                std::cout << "random samplig score " << r_score->inlier_number << "\n";                
                //
                // exit (0);
            }
//            std::cout << "Ransac, iteration " << iters << "; score " << current_score->inlier_number << "\n";
//            std::cout << models[i]->returnDescriptor() << "\n\n";

            if (current_score->bigger(best_score)) {

//                  std::cout << "current score = " << current_score->score << '\n';

                // if (current_score->inlier_number > min_inlier_count_for_LO) {
                //     irls->getModelScore(current_score, models[i]);
                // }
                
                // update current model and current score by inner and iterative local optimization
                // if inlier number is too small, do not update
                if (LO && current_score->inlier_number > min_inlier_count_for_LO) {
//                    std::cout << "score before LO " << current_score->inlier_number << "\n";
                    lo_ransac->GetLOModelScore (models[i], current_score);
//                    std::cout << "score after LO " << current_score->inlier_number << "\n";
                }

               // todo: termination conditions at first

                // update current model and current score by graph cut local optimization
                // if inlier number is too small, do not update
                if (GraphCutLO && current_score->inlier_number > min_inlier_count_for_LO) {
//                    std::cout << "score before GC LO " << current_score->inlier_number << "\n";
                    graphCut->GraphCutLO(models[i], current_score);
//                    std::cout << "score after GC LO " << current_score->inlier_number << "\n";
                    gc_runs++;
                }

                // copy current score to best score
                best_score->copyFrom(current_score);

                // remember best model
                best_model->setDescriptor (models[i]->returnDescriptor());

//                  std::cout << "Ransac, update best score" << best_score->inlier_number << '\n';

                // Termination conditions:
                if (is_prosac) {
                    max_iters = prosac_termination_criteria->
                            getUpBoundIterations(iters, prosac_sampler->getLargestSampleSize(),
                                                 best_model->returnDescriptor());
                } else {
                    max_iters = termination_criteria->getUpBoundIterations (best_score->inlier_number);
                }
                if (SprtLO) {
//                     std::cout << "std = " << max_iters << " vs sprt = " << sprt->getUpperBoundIterations(best_score->inlier_number) << "\n";
//                    std::cout << "sprt (usac) = " << sprt->updateSPRTStopping(best_score->inlier_number) << "\n";
                    max_iters = std::min (max_iters, (int)sprt->getUpperBoundIterations(best_score->inlier_number));
//                    std::cout << "got max iters \n";
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
//        exit (111);
    }

    // Graph Cut lo was set, but did not run, run it
    if (GraphCutLO && graphCut->gc_iterations == 0) {
        // update best model and best score
        graphCut->GraphCutLO(best_model, best_score);
    }

//    std::cout << "Calculate Non minimal model\n";

    Model *non_minimal_model = new Model (model);

//    std::cout << "end best inl num " << best_score->inlier_number << '\n';

    unsigned int previous_non_minimal_num_inlier = 0;

    int * max_inliers = new int[points_size];
    // get inliers from best model
    quality->getInliers(best_model->returnDescriptor(), max_inliers);

    for (unsigned int norm = 0; norm < 5 /* normalizations count */; norm++) {
        /*
         * TODO:
         * Calculate and Save Covariance Matrix and use it next normalization with adding or
         * extracting some points.
         */
//        std::cout << "end estimate non minimal\n";

        // estimate non minimal model with max inliers
        if (estimator->EstimateModelNonMinimalSample(max_inliers, best_score->inlier_number, *non_minimal_model)) {

//            std::cout << "end get non minimal score\n";

            quality->getNumberInliers(current_score, non_minimal_model, true, max_inliers);

            // Priority is for non minimal model estimation
//            std::cout << "non minimal inlier number " << current_score->inlier_number << '\n';

            // break if non minimal model score is less than 80% of the best minimal model score
            if ((float) current_score->inlier_number / best_score->inlier_number < 0.8) {
                break;
//                std::cout << "|I|best = " << best_score->inlier_number << "\n";
//                std::cout << "|I|non minimal = " << current_score->inlier_number << "\n";
//                std::cout << "\033[1;31mNON minimal model has less than 50% of inliers to compare with best score!\033[0m \n";
            }

            // if normalization score is less or equal, so next normalization is equal too, so break.
            if (current_score->inlier_number <= previous_non_minimal_num_inlier) {
                break;
            }

            previous_non_minimal_num_inlier = current_score->inlier_number;

            best_score->copyFrom(current_score);
            best_model->setDescriptor(non_minimal_model->returnDescriptor());
        } else {
            std::cout << "\033[1;31mNON minimal model completely failed!\033[0m \n";
            break;
        }
    }

    std::chrono::duration<float> fs = std::chrono::steady_clock::now() - begin_time;
    // ================= here is ending ransac main implementation ===========================

    // get final inliers from the best model
    quality->getInliers(best_model->returnDescriptor(), max_inliers);
//    std::cout << "FINAL best inl num " << best_score->inlier_number << '\n';
//    std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";

    unsigned int lo_inner_iters = 0;
    unsigned int lo_iterative_iters = 0;
    if (LO) {
        RansacLocalOptimization * lo_r = (RansacLocalOptimization *) lo_ransac;
        lo_inner_iters = lo_r->lo_inner_iters;
        lo_iterative_iters = lo_r->lo_iterative_iters;
    }
    unsigned int gc_iters = 0;
    if (GraphCutLO) {
        gc_iters = graphCut->gc_iterations;
    }
    // Store results
    ransac_output = new RansacOutput (best_model, max_inliers,
            std::chrono::duration_cast<std::chrono::microseconds>(fs).count(),
                                      best_score->inlier_number, iters, lo_inner_iters, lo_iterative_iters, gc_iters);

    if (LO) {
        delete[] lo_ransac;
    }
    if (GraphCutLO) {
        delete[] graphCut;
    }
    if (SprtLO) {
        delete[] sprt;
    }
    delete[] sample;
    delete[] current_score;
    delete[] best_score;
    delete[] inliers;
    delete[] max_inliers;
//    delete[] best_model;
}








//            Drawing drawing;
//            cv::Mat img1 = cv::imread ("../dataset/homography/LePoint1A.png");
//            cv::Mat img2 = cv::imread ("../dataset/homography/LePoint1B.png");
//            cv::Mat pts1 = input_points.getMat().colRange (0, 2);
//            cv::Mat pts2 = input_points.getMat().colRange (2, 4);
//            cv::hconcat (pts1, cv::Mat_<float>::ones(points_size, 1), pts1);
//            cv::hconcat (pts2, cv::Mat_<float>::ones(points_size, 1), pts2);
//            cv::Mat pt11 = pts1.row (sample[0]);
//            cv::Mat pt12 = pts1.row (sample[1]);
//            cv::Mat pt13 = pts1.row (sample[2]);
//            cv::Mat pt14 = pts1.row (sample[3]);
//
//            cv::Mat pt21 = pts2.row (sample[0]);
//            cv::Mat pt22 = pts2.row (sample[1]);
//            cv::Mat pt23 = pts2.row (sample[2]);
//            cv::Mat pt24 = pts2.row (sample[3]);
//
//            std::cout << sample[0] << " " << sample[1] << " " << sample[2] << " " << sample[3] << "\n";
//            std::cout << pts1.row(sample[0]) << "\n" << pts1.row(sample[1]) << "\n" << pts1.row(sample[2]) << "\n" <<pts1.row(sample[3]) << "\n";
//            std::cout << models[i]->returnDescriptor() << "\n";
//            std::cout << "-----------------------------\n";
//            drawing.drawErrors (img1, img2, pts1, pts2, models[i]->returnDescriptor());
//            cv::circle (img1, cv::Point_<float>(pts1.at<float>(sample[0], 0), pts1.at<float>(sample[0], 1)), 7, cv::Scalar(255, 255, 0), -1);
//            cv::circle (img1, cv::Point_<float>(pts1.at<float>(sample[1], 0), pts1.at<float>(sample[1], 1)), 7, cv::Scalar(255, 100, 0), -1);
//            cv::circle (img1, cv::Point_<float>(pts1.at<float>(sample[2], 0), pts1.at<float>(sample[2], 1)), 7, cv::Scalar(100, 255, 0), -1);
//            cv::circle (img1, cv::Point_<float>(pts1.at<float>(sample[3], 0), pts1.at<float>(sample[3], 1)), 7, cv::Scalar(255, 255, 255), -1);
//
//            cv::circle (img2, cv::Point_<float>(pts2.at<float>(sample[0], 0), pts2.at<float>(sample[0], 1)), 7, cv::Scalar(255, 255, 0), -1);
//            cv::circle (img2, cv::Point_<float>(pts2.at<float>(sample[1], 0), pts2.at<float>(sample[1], 1)), 7, cv::Scalar(255, 100, 0), -1);
//            cv::circle (img2, cv::Point_<float>(pts2.at<float>(sample[2], 0), pts2.at<float>(sample[2], 1)), 7, cv::Scalar(100, 255, 0), -1);
//            cv::circle (img2, cv::Point_<float>(pts2.at<float>(sample[3], 0), pts2.at<float>(sample[3], 1)), 7, cv::Scalar(255, 255, 255), -1);
//
//            cv::hconcat (img1, img2, img1);
//            cv::imshow ("homography", img1);
//            cv::waitKey(0);





// ---------- for debug ----------------------
//            Drawing drawing;
//            cv::Mat img = cv::imread ("../dataset/image1.jpg");
//            // std::vector<int> inl;
//            // for (int i = 0; i < lo_score->inlier_number; i++) {
//            //     inl.push_back(lo_inliers[i]);
//            // }
//            // drawing.showInliers(input_points, inl, img);
//            cv::Point_<float> * pts = (cv::Point_<float> *) input_points.getMat().data;
//            drawing.draw_model(models[0], cv::Scalar(255, 0, 0), img, false);
//            cv::circle (img, pts[sample[0]], 3, cv::Scalar(255, 255, 0), -1);
//            cv::circle (img, pts[sample[1]], 3, cv::Scalar(255, 255, 0), -1);
//            cv::imshow("samples img", img); cv::waitKey(0);
//            cv::imwrite( "../results/"+model->model_name+"_"+std::to_string(iters)+".jpg", img);
// -------------------------------------------
//            Drawing drawing;
//            cv::Mat img = cv::imread ("../dataset/homography/boatA.png");
//            // std::vector<int> inl;
//            // for (int i = 0; i < lo_score->inlier_number; i++) {
//            //     inl.push_back(lo_inliers[i]);
//            // }
//            // drawing.showInliers(input_points, inl, img);
//            cv::imshow("H", img); cv::waitKey(0);
// -------------------------------------------



//     Drawing drawing;
//     cv::Mat img1 = cv::imread ("../dataset/homography/LePoint1A.png");
//     cv::Mat img2 = cv::imread ("../dataset/homography/LePoint1B.png");
//     cv::Mat pts1 = input_points.getMat().colRange (0, 2);
//     cv::Mat pts2 = input_points.getMat().colRange (2, 4);
//     cv::hconcat (pts1, cv::Mat_<float>::ones(points_size, 1), pts1);
//     cv::hconcat (pts2, cv::Mat_<float>::ones(points_size, 1), pts2);
//     drawing.drawErrors (img1, img2, pts1, pts2, best_model->returnDescriptor());

// quality->getInliers(best_model->returnDescriptor(), inliers);
// for (int i = 0; i < best_score->inlier_number; i++) {
//    cv::circle (img1, cv::Point_<float>(pts1.at<float>(inliers[i], 0), pts1.at<float>(inliers[i], 1)), 4, cv::Scalar(255, 255, 255), -1);
// }
// cv::hconcat (img1, img2, img1);
//     cv::imshow ("homography", img1);
//     cv::imwrite("../results/homography/lepoint1.png", img1);
//     cv::waitKey(0);

