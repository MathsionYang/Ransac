#ifndef USAC_LMS_QUALITY_H
#define USAC_LMS_QUALITY_H

#include "quality.hpp"
#include "../utils/utils.hpp"

class LmsScore : public Score {
public:
    LmsScore () {
        score = FLT_MAX;
    }

    // priority to score
    bool better(const Score * const score2) override {
        return score < score2->score;
    }
    bool better(const Score &score2) override {
        return score < score2.score;
    }
};

class LmsQuality : public Quality {
private:
    float * errors;
public:
    ~LmsQuality() override {
        delete[] errors;
    }

    void init2 () {
        // note Quality::init(...) must be called prior
        errors = new float[points_size];
    }
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
            errors[point] = err;
            if (err < threshold) {
                if (get_inliers) {
                    inliers[inlier_number] = point;
                }
                inlier_number++;
            }
        }

        score->score = findMedian (errors, points_size);
        score->inlier_number = inlier_number;

        // debug
        std::cout  << "median score " << score->score << "\n";

        std::sort (errors, errors + points_size);
        if (points_size % 2) {
            // odd number
            std::cout << "GT median " << errors[points_size / 2] << "\n";
        } else {
            // even
            std::cout << "GT median " << (errors[points_size / 2] + errors[points_size / 2 - 1]) / 2 << "\n";
        }

    }
};

#endif //USAC_LMS_QUALITY_H
