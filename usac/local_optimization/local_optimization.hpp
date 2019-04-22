#ifndef USAC_LOCALOPTIMIZATION_H
#define USAC_LOCALOPTIMIZATION_H

#include "../precomp.hpp"
#include "../estimator/estimator.hpp"
#include "../quality/quality.hpp"

class LocalOptimization {
public:
	virtual ~LocalOptimization () = default;

	/*
	 * Update best model and best score.
	 */
    virtual void GetModelScore  (Model * best_model, Score * best_score) = 0;
	virtual unsigned int getNumberIterations () = 0;
};


#endif //USAC_LOCALOPTIMIZATION_H
