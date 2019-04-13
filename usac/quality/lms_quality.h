#ifndef USAC_LMS_QUALITY_H
#define USAC_LMS_QUALITY_H

#include "quality.hpp"

class LMSScore : public Score {
public:

};

class LMSQuality : public Quality {
public:

    void getScore (Score * score, const cv::Mat& model, float threshold, bool get_inliers,
                           int * inliers, bool parallel) override {

    }
};

#endif //USAC_LMS_QUALITY_H
