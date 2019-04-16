#ifndef USAC_MLESAC_QUALITY_H
#define USAC_MLESAC_QUALITY_H

// https://github.com/stevewen/PR2_door_open/blob/205a86ce73abf34346fc1ddf31bb232e2d716c81/door_handle_detector/src/sample_consensus/mlesac.cpp
//http://www.robots.ox.ac.uk/~vgg/publications-new/Public/2000/Torr00/torr00.pdf

class MlesacScore : public Score {
public:
    MlesacScore () {
        score = FLT_MAX;
    }

    // compare -log likelihood, if less than better
    bool better(const Score * const score2) override {
        return score < score2->score;
    }
    bool better(const Score &score2) override {
        return score < score2.score;
    }
};

class MlesacQuality : public Quality {
private:
    // according to Torr 2-3 iterations required EM to converge.
    unsigned int iterations_EM = 3;
    // Initial estimate for the gamma mixing parameter = 1/2
    // sigma - standard deviation of error on each coordinate
    float gamma = 0.5, sigma, v;
    float * inlier_prob, * errors;
public:
    ~MlesacQuality() override {
        delete[] inlier_prob;
        delete[] errors;
    }

    // note Quality::init() must be called before
    void init2 (const int * const points) {
        sigma = 0; // todo
        v = 0; // todo
//        sqrt ( (max_pt.x - min_pt.x) * (max_pt.x - min_pt.x) +
//                          (max_pt.y - min_pt.y) * (max_pt.y - min_pt.y) +
//                          (max_pt.z - min_pt.z) * (max_pt.z - min_pt.z)
//        );
        inlier_prob = new float[points_size];
        errors = new float[points_size];
    }

    void getScore (Score * score, const cv::Mat& model, float threshold, bool get_inliers,
                   int * inliers, bool parallel) override {

        if (threshold == 0) {
            threshold = this->threshold;
        }

        estimator->setModelParameters(model);

        float p_outlier_prob, err;
        unsigned int inlier_number = 0;
        bool find_inliers = true;

        for (int j = 0; j < iterations_EM; j++) {
            // Likelihood of a datum given that it is an inlier
            for (unsigned int i = 0; i < points_size; i++) {
                if (find_inliers) {
                    err = estimator->GetError(i);
                    errors[i] = err;
                    if (err < threshold) {
                        if (get_inliers) {
                            inliers[inlier_number] = i;
                        }
                        inlier_number++;
                    }
                }
                inlier_prob[i] = gamma * exp(-(errors[i] * errors[i]) / 2 * (sigma * sigma)) / (sqrt(2 * M_PI) * sigma);
            }
            find_inliers = false;

            // Likelihood of a datum given that it is an outlier
            p_outlier_prob = (1 - gamma) / v;

            // Use Expectiation-Maximization to find out the right value for d_cur_penalty
            gamma = 0;
            for (unsigned int i = 0; i < points_size; i++) {
                gamma += inlier_prob[i] / (inlier_prob[i] + p_outlier_prob);
            }
            gamma /= points_size;
        }

        // Find the log likelihood of the model -L = -sum [log (pInlierProb + pOutlierProb)]
        float L = 0;
        for (unsigned int i = 0; i < points_size; i++)
            L += log (inlier_prob[i] + p_outlier_prob);
        L = - L;

        score->score = L;
        score->inlier_number = inlier_number;
    }
};

#endif //USAC_MLESAC_QUALITY_H
