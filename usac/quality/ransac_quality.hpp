#ifndef RANSAC_QUALITY_H
#define RANSAC_QUALITY_H

#include "../precomp.hpp"

#include "../estimator/estimator.hpp"
#include "../model.hpp"
#include "quality.hpp"

class RansacScore : public Score {
public:
    // priority to inlier number
    bool better(const Score * const score2) override {
        return inlier_number > score2->inlier_number;
    }
    bool better(const Score &score2) override {
        return inlier_number > score2.inlier_number;
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

        if (parallel && !get_inliers) {
            #pragma omp parallel for reduction (+:inlier_number)
            for (unsigned int point = 0; point < points_size; point++) {
                if (estimator->GetError(point) < threshold) {
                    inlier_number++;
                }
            }
        } else {
            if (get_inliers) {
                for (unsigned int point = 0; point < points_size; point++) {
                    if (estimator->GetError(point) < threshold) {
                        inliers[inlier_number++] = point;
                    }
                }
            } else {
                for (unsigned int point = 0; point < points_size; point++) {
                    if (estimator->GetError(point) < threshold) {
                        inlier_number++;
                    }
                }
            }
        }

        score->inlier_number = inlier_number;
        score->score = inlier_number;
    }
};


#endif //USAC_RANSACQUALITY_H
