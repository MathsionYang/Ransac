#ifndef USAC_MSAC_QUALITY_H
#define USAC_MSAC_QUALITY_H

#include "quality.hpp"

class MsacScore : public Score {
public:
    MsacScore () {
        score = FLT_MAX;
    }

    // priority to score, if score is less than better, smaller residuals
    bool better(const Score * const score2) override {
        return score < score2->score;
    }
    bool better(const Score &score2) override {
        return score < score2.score;
    }
};

class MsacQuality : public Quality {
public:

    void getScore (Score * score, const cv::Mat& model, float threshold, bool get_inliers,
                   int * inliers, bool parallel) override {

        if (threshold == 0) {
            threshold = this->threshold;
        }

        estimator->setModelParameters(model);

        float err, sum_errors = 0;
        unsigned int inlier_number = 0;
        for (unsigned int point = 0; point < points_size; point++) {
            err = estimator->GetError(point);
            if (err < threshold) {
                if (get_inliers) {
                    inliers[inlier_number] = point;
                }
                inlier_number++;
                sum_errors += err;
            } else {
                sum_errors += threshold;
            }
        }

        score->score = sum_errors;
        score->inlier_number = inlier_number;
    }
};


#endif //USAC_MSAC_QUALITY_H
