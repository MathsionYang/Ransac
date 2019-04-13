#ifndef RANSAC_QUALITY_H
#define RANSAC_QUALITY_H

#include "../precomp.hpp"

#include "../estimator/estimator.hpp"
#include "../model.hpp"
#include "quality.hpp"

class RansacScore : public Score {
public:
    // priority for inlier number
    bool bigger (const Score * const score2) override {
        if (inlier_number > score2->inlier_number) return true;
        if (inlier_number == score2->inlier_number) return score > score2->score;
        return false;
    }
    bool bigger (const Score& score2) override {
        if (inlier_number > score2.inlier_number) return true;
        if (inlier_number == score2.inlier_number) return score > score2.score;
        return false;
    }

    void copyFrom (const Score * const score_to_copy) override {
        score = score_to_copy->score;
        inlier_number = score_to_copy->inlier_number;
    }

    void copyFrom (const Score &score_to_copy) override {
        score = score_to_copy.score;
        inlier_number = score_to_copy.inlier_number;
    }
};

class RansacQuality : public Quality {
public:

    /*
     * calculating number of inliers of current model.
     * score is sum of distances to estimated inliers.
     */
    // use inline
    void getScore (Score * score, const cv::Mat& model, float threshold, bool get_inliers,
                                  int * inliers, bool parallel) override {
        if (threshold == 0) {
            threshold = this->threshold;
        }

        estimator->setModelParameters(model);

        unsigned int inlier_number = 0;
        float sum_errors = 0;

        if (parallel && !get_inliers) {
            #pragma omp parallel for reduction (+:inlier_number)
            for (unsigned int point = 0; point < points_size; point++) {
                if (estimator->GetError(point) < threshold) {
                    inlier_number++;
                }
            }
        } else {
            // inlier_number = estimator->GetNumInliers(threshold, get_inliers, inliers);
            float err;
            if (get_inliers) {
                for (unsigned int point = 0; point < points_size; point++) {
                    err = estimator->GetError(point);
                    if (err < threshold) {
                        inliers[inlier_number++] = point;
                        sum_errors += err;
                    }
                }
            } else {
                for (unsigned int point = 0; point < points_size; point++) {
                    err = estimator->GetError(point);
                    if (err < threshold) {
                        inlier_number++;
                        sum_errors += err;
                    }
                }
            }
        }

        score->inlier_number = inlier_number;
        score->score = sum_errors;
    }
};


#endif //USAC_RANSACQUALITY_H
