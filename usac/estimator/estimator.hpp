#ifndef RANSAC_ESTIMATOR_H
#define RANSAC_ESTIMATOR_H

#include "../precomp.hpp"

#include "../sampler/sampler.hpp"
#include "../model.hpp"
#include "../termination_criteria/standard_termination_criteria.hpp"

class Estimator {
public:
    virtual ~Estimator () = default;

    // minimal model estimation
    virtual unsigned int EstimateModel(const int * const sample, std::vector<Model>& models) = 0;
    
    virtual bool EstimateModelNonMinimalSample(const int * const sample, unsigned int sample_size, Model &model) = 0;
    virtual bool LeastSquaresFitting (const int * const sample, unsigned int sample_size, Model &model) {
        return EstimateModelNonMinimalSample(sample, sample_size, model);
    }

    virtual bool EstimateModelNonMinimalSample(const int * const sample, unsigned int sample_size, const float * const weights, Model &model) {
        std::cout << "NOT IMPLEMENTED EstimateModelNonMinimalSample in estimator\n";
    }
    virtual bool EstimateModelNonMinimalSample(const int * const sample, unsigned int sample_size, const float * const weightsx, const float * const weightsy, Model &model) {
        std::cout << "NOT IMPLEMENTED EstimateModelNonMinimalSample in estimator\n";
    }
    virtual bool EstimateModelNonMinimalSample(const int * const sample, unsigned int sample_size,
            const float * const weightsx1, const float * const weightsy1,
            const float * const weightsx2, const float * const weightsy2, Model &model) {
        std::cout << "NOT IMPLEMENTED EstimateModelNonMinimalSample in estimator\n";
    }

    virtual unsigned int getInliersWeights (float threshold,
                             int * inliers,
                             bool get_error, float * errors,
                             bool get_euc, float * weights_euc1,
                             bool get_euc2, float * weights_euc2,
                             bool sampson, float * weights_sampson,
                             bool get_manh, float * weights_manh1,
                             float * weights_manh2,
                             float * weights_manh3,
                             float * weights_manh4) {}

    virtual bool isSubsetGood (const int * const sample) {
        return true;
    }

    // use inline
    virtual float GetError(unsigned int pidx) = 0;
    virtual unsigned int GetNumInliers (float threshold, bool get_inliers=false, int * inliers=nullptr) { return 0; }

    virtual int SampleNumber() = 0;

    virtual void setModelParameters (const cv::Mat& model) = 0;

    virtual bool isModelValid (const cv::Mat &model, const int * const sample) {
        return true;
    }
};

#endif //RANSAC_ESTIMATOR_H