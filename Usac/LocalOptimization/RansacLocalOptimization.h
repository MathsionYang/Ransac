#ifndef USAC_RANSACLOCALOPTIMIZATION_H
#define USAC_RANSACLOCALOPTIMIZATION_H

#include "LocalOptimization.h"
#include "../Quality.h"
#include "../Helper/Drawing/Drawing.h"

/*
 * Reference:
 * http://cmp.felk.cvut.cz/~matas/papers/chum-dagm03.pdf
 */

class RansacLocalOptimization : public LocalOptimization {
private:
    Model * model;
    Quality * quality;
    TerminationCriteria * termination_criteria;
    Sampler *sampler;
    Estimator * estimator;
    
public:

    RansacLocalOptimization (Model * model_,
                             Sampler *sampler_,
                             TerminationCriteria *termination_criteria_,
                             Quality *quality_,
                             Estimator *estimator_) {
        model = model_;
        sampler = sampler_;
        estimator = estimator_;
        termination_criteria = termination_criteria_;
        quality = quality_;
    }
    
    bool GetLOModelScore (Model &best_lo_model,
                          Score &best_lo_score,
                          Score *kth_ransac_score,
                          cv::InputArray input_points,
                          unsigned int points_size,
                          unsigned int kth_step,
                          const int * const inliers) override {

        /*
         * Do not do local optimization for small number of inliers
         */
        if (kth_ransac_score->inlier_number < 2 * model->lo_sample_size) {
            return false;
        }

        /*
         * Let's best lo score is at least kth ransac score.
         * Avoid comparing between current lo score with best score and kth score.
         */
        best_lo_score.inlier_number = kth_ransac_score->inlier_number;
        best_lo_score.score = kth_ransac_score->score;

        /* In our experiments the size of samples are set to min (Ik/2, 14)
         * for epipolar geometry and to min (Ik/2, 12) for the case of homography estimation
         */
        unsigned int lo_sample_size = std::min(kth_ransac_score->inlier_number / 2 + 1, (int) best_lo_model.lo_sample_size);
        unsigned int lo_max_iterations = best_lo_model.lo_max_iterations;
        unsigned int lo_iterative_iterations = best_lo_model.lo_iterative_iterations;

        // std::cout << "lo sample_size " << lo_sample_size << '\n';
            

        Model *lo_model = new Model(*model);
        Score *lo_score = new Score;
        
        int *lo_sample = new int[lo_sample_size];
        int * lo_inliers = new int[points_size];

        sampler->setSampleSize(lo_sample_size);
        sampler->setPointsSize(kth_ransac_score->inlier_number);
        sampler->initRandomGenerator();

        /*
         * reduce multiplier threshold K·θ by this number in each iteration.
         * In the last iteration there be original threshold θ.
         */
        float reduce_threshold = (lo_model->lo_threshold * lo_model->lo_threshold_multiplier -
                                  lo_model->threshold)
                                 / lo_model->lo_iterative_iterations;

        bool lo_better_than_kth_ransac = false;
        /*
         * Inner Local Optimization Ransac
         */
        unsigned int max_iters = kth_step + lo_max_iterations;
        unsigned int lo_iters = kth_step;
        for (int iters = 0; iters < lo_max_iterations; iters++) {
            if (lo_iters > max_iters) {
                return true;
            }

            /*
             * Generate sample of lo_sample_size from reached best_sample in current iteration
             */
            sampler->generateSample(lo_sample);
            if (lo_better_than_kth_ransac) {
                for (int smpl = 0; smpl < lo_sample_size; smpl++) {
                    lo_sample[smpl] = lo_inliers[lo_sample[smpl]];
                }   
            } else {
                for (int smpl = 0; smpl < lo_sample_size; smpl++) {
                    lo_sample[smpl] = inliers[lo_sample[smpl]];
                }
            }


            /*
             * Estimate model of best sample from k-th step of Ransac
             */
            if (!estimator->EstimateModelNonMinimalSample(lo_sample, lo_sample_size, *lo_model))
                continue;
            

            estimator->setModelParameters(lo_model);

            // Evaluate model and get inliers
            quality->GetModelScore(estimator, lo_model, input_points, points_size, *lo_score, lo_inliers, true);

            /*
             * If current inner lo score worse than best lo score, so
             * current inliers for iterative lo score are inliers from best ransac score
             * Else best lo score is lo score;
             */

            if (*lo_score > best_lo_score) {
                best_lo_score.copyFrom (lo_score);
                best_lo_model.setDescriptor(lo_model->returnDescriptor());
            }

            std::cout << "lo inner score = " << lo_score->score << '\n';

            lo_model->threshold = lo_model->lo_threshold_multiplier * lo_model->lo_threshold;

            /*
             * Iterative LO Ransac
             * Reduce threshold of current model
             * Estimate model parametres
             * Evaluate model
             * Get inliers
             * Repeat until iteration < lo_iterations
             */

            Drawing drawing;
            cv::Scalar color = cv::Scalar (random()%256,random()%256,random()%256);

            for (int iterations = 0; iterations < lo_iterative_iterations; iterations++) {
                lo_model->threshold -= reduce_threshold;

                // std::cout << "lo  threshold " << lo_model->threshold << '\n';
//                std::cout << "begin lo score " << lo_score->inlier_number << '\n';

                estimator->LeastSquaresFitting(lo_inliers, lo_score->inlier_number, *lo_model);
                estimator->setModelParameters(lo_model);

                quality->GetModelScore(estimator, lo_model, input_points, points_size, *lo_score, lo_inliers, true);

                // std::cout << "lo iterative score  = " << lo_score->inlier_number << '\n';
//                std::cout << "lo model  = " << lo_model->returnDescriptor() << '\n';


                // ---------- for debug ----------------------
                // Drawing drawing;
                // cv::Mat img = cv::imread ("../dataset/image1.jpg");
                // int rows = img.rows;
                // int cols = img.cols;
                // std::vector<int> inl;
                // for (int i = 0; i < lo_score->inlier_number; i++) {
                //     inl.push_back(lo_inliers[i]);
                // }
                // drawing.showInliers(input_points, inl, img);
                // drawing.draw_model(lo_model, std::max(rows, cols), color, img, true);
                // cv::imshow("least img", img);  
                // cv::waitKey(0);
                // -------------------------------------------

                lo_iters++;
                // if current model is not better then break
                if (best_lo_score > lo_score) {
                    break;
                }

            }

            // std::cout << "end iterative lo inner score  = " << lo_score->inlier_number << '\n';

            if (*lo_score > best_lo_score) {
                best_lo_model.setDescriptor(lo_model->returnDescriptor());
                best_lo_score.copyFrom (lo_score);
                lo_better_than_kth_ransac = true;

                max_iters = termination_criteria->getUpBoundIterations(best_lo_score.inlier_number, points_size);
            }
            lo_iters++;
        }

        sampler->setSampleSize(model->sample_number);
        sampler->setPointsSize(points_size);
        sampler->initRandomGenerator();

        return false;
    }
};


#endif //USAC_RANSACLOCALOPTIMIZATION_H
