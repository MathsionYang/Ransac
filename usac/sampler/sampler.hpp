#ifndef RANSAC_SAMPLER_H
#define RANSAC_SAMPLER_H

#include "../precomp.hpp"
#include "../random_generator/random_generator.hpp"

class Sampler {
protected:
    unsigned int k_iterations = 0, points_size = 0, sample_size = 0;
public:
    virtual ~Sampler() = default;

    /*
     * generate sample. Considering that all parameters are defined (including sample_size, points_size and
     * random generator)
     */
    virtual void generateSample (int *sample) = 0;

    /*
     * Returns number of iterations
     */
    unsigned int getNumberOfIterations () {
        return k_iterations;
    }

    /*
     * Check if sampler is safe to use and everything is initialized
     * Can be different for child classes
     */
    virtual bool isInit ()  { return false; }
};

#endif //RANSAC_SAMPLER_H