#ifndef QUALITY_H
#define QUALITY_H

#include "../precomp.hpp"

#include "../estimator/estimator.hpp"
#include "../model.hpp"

class Score {
public:
    virtual ~Score() = default;
    unsigned int inlier_number = 0;
    float score = 0;

    virtual bool better(const Score * const score2) = 0;
    virtual bool better(const Score &score2) = 0;

    // use inline
    void copyFrom (const Score * const score_to_copy) {
        score = score_to_copy->score;
        inlier_number = score_to_copy->inlier_number;
    }

    void copyFrom (const Score &score_to_copy) {
        score = score_to_copy.score;
        inlier_number = score_to_copy.inlier_number;
    }
};

class Quality {
protected:
    unsigned int points_size;
    float threshold;
    Estimator * estimator;
    bool isinit = false;
public:
    virtual ~Quality() = default;

    virtual bool isInit () { return isinit; }

    void init (unsigned int points_size_, float threshold_, Estimator * estimator_) {
        points_size = points_size_;
        threshold = threshold_;
        estimator = estimator_;
        isinit = true;
    }

    /*
     * calculating number of inliers of current model.
     * score is sum of distances to estimated inliers.
     */
    virtual void getScore (Score * score, const cv::Mat& model, float threshold=0, bool get_inliers=false,
                                  int * inliers= nullptr, bool parallel=false) = 0;

    void getInliers (const cv::Mat& model, int * inliers) {
        // Note class Quality should be initialized
        assert(isinit);

        estimator->setModelParameters(model);

	    int num_inliers = 0;
	    for (unsigned int point = 0; point < points_size; point++) {
            if (estimator->GetError(point) < threshold) {
                inliers[num_inliers] = point;
                num_inliers++;
            }
        }
    }

    void getInliers (const cv::Mat& model, int * inliers, float threshold) {
        // Note class Quality should be initialized
        assert(isinit);

        estimator->setModelParameters(model);

        int num_inliers = 0;
        for (unsigned int point = 0; point < points_size; point++) {
            if (estimator->GetError(point) < threshold) {
                inliers[num_inliers] = point;
                num_inliers++;
            }
        }
    }


    static void getInliers (Estimator * estimator, const cv::Mat &model, float threshold, unsigned int points_size, std::vector<int> &inliers) {
        estimator->setModelParameters(model);
        inliers.clear();
        for (unsigned int p = 0; p < points_size; p++) {
            if (estimator->GetError(p) < threshold) {
                inliers.push_back(p);
            }
        }
    }

    // Get average error to GT inliers.
    static float getErrorToGTInliers(Estimator *estimator, const cv::Mat &model, const std::vector<int> &gt_inliers) {

        // return -1 (unknown) to avoid division by zero
        if (gt_inliers.size() == 0)
            return -1;

        float sum_errors = 0;
        estimator->setModelParameters(model);
        for (unsigned int inl : gt_inliers) {
            sum_errors += estimator->GetError(inl);
        }
        return sum_errors / gt_inliers.size();
    }
};


#endif //QUALITY_H