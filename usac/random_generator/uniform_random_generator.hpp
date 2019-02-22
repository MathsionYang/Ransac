#ifndef USAC_UNIFORMRANDOMGENERATOR_H
#define USAC_UNIFORMRANDOMGENERATOR_H

#include "random_generator.hpp"

class UniformRandomGenerator : public RandomGenerator {
protected:
    std::mt19937 generator;
    std::uniform_int_distribution<int> generate;
public:
    UniformRandomGenerator () {
        std::random_device rand_dev;
        generator = std::mt19937(rand_dev());
    }

    int getRandomNumber () override {
        return generate (generator);
    }

    void resetGenerator (int min_range, int max_range) override {
        generate = std::uniform_int_distribution<int>(min_range, max_range);
    }

    void generateUniqueRandomSet (int * sample) override {
        int num, j;
        sample[0] = generate (generator);
        for (unsigned int i = 1; i < subset_size;) {
            num = generate (generator);
            for (j = i - 1; j >= 0; j--) {
                if (num == sample[j]) {
                    break;
                }
            }
            if (j == -1) sample[i++] = num;
        }
    }

    void generateUniqueRandomSet (int * sample, unsigned int subset_size, unsigned int max) {
        assert(subset_size+1 <= max);
        resetGenerator(0, max);
        int num, j;
        sample[0] = generate (generator);
        for (unsigned int i = 1; i < subset_size;) {
            num = generate (generator);
            for (j = i - 1; j >= 0; j--) {
                if (num == sample[j]) {
                    break;
                }
            }
            if (j == -1) sample[i++] = num;
        }
    }

    // closed interval <0; max>
    void generateUniqueRandomSet (int * sample, unsigned int max) {
        assert(subset_size+1 <= max);
        resetGenerator(0, max);
        int num, j;
        sample[0] = generate (generator);
        for (unsigned int i = 1; i < subset_size;) {
            num = generate (generator);
            for (j = i - 1; j >= 0; j--) {
                if (num == sample[j]) {
                    break;
                }
            }
            if (j == -1) sample[i++] = num;
        }
    }

};



#endif //USAC_UNIFORMRANDOMGENERATOR_H
