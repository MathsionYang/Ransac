#ifndef RANSAC_MODEL_H
#define RANSAC_MODEL_H

#include "../dataset/Dataset.h"

enum ESTIMATOR  { Line2d, Homography, Fundamental, Essential, Affine };
enum SAMPLER  { Uniform, ProgressiveNAPSAC, Napsac, Prosac, Evsac, ProsacNapsac };
enum NeighborsSearch {NullN, Nanoflann, Grid};
enum LocOpt {NullLO, InLORsc, ItLORsc, ItFLORsc, InItLORsc, InItFLORsc, GC, IRLS};
enum SCORE {RANSAC, MSAC, LMS, MLESAC};

class Model {
public:
	float threshold = 2;
	float confidence = 0.95;

    unsigned int sample_size;
    unsigned int min_iterations = 20;
    unsigned int max_iterations = 5000;
	unsigned int k_nearest_neighbors = 5;

	// Local Optimization parameters
	LocOpt lo = NullLO;
    unsigned int lo_sample_size = 12;
	unsigned int lo_iterative_iterations = 4;
    unsigned int lo_inner_iterations = 10; // 20
    unsigned int lo_threshold_multiplier = 10;

    // Graph cut
    float spatial_coherence_gc = 0.1; // spatial coherence term

    ESTIMATOR estimator;
    SAMPLER sampler;
    SCORE score = SCORE ::RANSAC;
    // sprt
    bool sprt = false;
	unsigned int max_hypothesis_test_before_sprt = 20;
	
    NeighborsSearch neighborsType = NeighborsSearch::NullN;
    int cell_size = 50; // for grid neighbors searching

    bool reset_random_generator = true;

    // for debug
    std::string img_name;
    DATASET dataset;
private:
    cv::Mat descriptor;
	
public:
	~Model () {}

    Model (const Model * const model) {
	    copyFrom(model);
	}

	Model (float threshold_, float desired_prob_, unsigned int knn,
		ESTIMATOR estimator_, SAMPLER sampler_) {
		if (estimator_ == ESTIMATOR::Line2d) sample_size = 2;
		else if (estimator_ == ESTIMATOR::Essential) sample_size = 5;
		else if (estimator_ == ESTIMATOR::Fundamental) {
			sample_size = 7;
			lo_sample_size = 14;
		}
		else if (estimator_ == ESTIMATOR::Homography) sample_size = 4;
		else if (estimator_ == ESTIMATOR::Affine) sample_size = 3;
		else {
			std::cout << "unexpected estimator!\n";
			exit (1);
		}

		threshold = threshold_;
		confidence = desired_prob_;
		k_nearest_neighbors = knn;
		estimator = estimator_;
		sampler = sampler_;
	}

	void setScore (SCORE score_){
	    score = score_;
	}

	void ResetRandomGenerator (bool reset) {
		reset_random_generator = reset;
	}

	void setNeighborsType (NeighborsSearch neighborsType_) {
        neighborsType = neighborsType_;
    }

    void setCellSize (int cell_size_) {
		cell_size = cell_size_;
	}

	void setSprt (bool SprtLO_) {
		sprt = SprtLO_;
	}

	void setLOParametres (unsigned int lo_iterative_iters, unsigned int lo_inner_iters, unsigned int lo_thresh_mult) {
	    lo_iterative_iterations = lo_iterative_iters;
	    lo_inner_iterations = lo_inner_iters;
        lo_threshold_multiplier = lo_thresh_mult;
	}

    inline void setDescriptor(const cv::Mat &desc) {
//    	descriptor = desc;
        desc.copyTo(descriptor);
    }

    cv::Mat returnDescriptor () {
        return descriptor;
    }

	void setThreshold (float threshold) {
		this->threshold = threshold;
	}

	void setDesiredProbability (float desired_prob) {
		this->confidence = desired_prob;
	}

	void setKNearestNeighbors (unsigned int k_nearest_neighbors) {
	    this->k_nearest_neighbors = k_nearest_neighbors;
	}

	void copyFrom (const Model * const model) {
        threshold = model->threshold;
        sample_size = model->sample_size;
        confidence = model->confidence;
        max_iterations = model->max_iterations;
		min_iterations = model->min_iterations;
		estimator = model->estimator;
        sampler = model->sampler;
        k_nearest_neighbors = model->k_nearest_neighbors;
        lo_sample_size = model->lo_sample_size;
        lo_iterative_iterations = model->lo_iterative_iterations;
        lo_inner_iterations = model->lo_inner_iterations;
        lo_threshold_multiplier = model->lo_threshold_multiplier;
        reset_random_generator = model->reset_random_generator;
        lo = model->lo;
        sprt = model->sprt;
        spatial_coherence_gc = model->spatial_coherence_gc;
        cell_size = model->cell_size;
        neighborsType = model->neighborsType;
        max_hypothesis_test_before_sprt = model->max_hypothesis_test_before_sprt;
        if (!model->descriptor.empty()) {
			model->descriptor.copyTo(descriptor);
		}
	}
};

#endif //RANSAC_MODEL_H