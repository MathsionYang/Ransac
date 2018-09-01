#ifndef USAC_PRIMENUMBERRANDOMGENERATOR_H
#define USAC_PRIMENUMBERRANDOMGENERATOR_H

#include "RandomGenerator.h"
#include <unordered_set>

// https://github.com/preshing/RandomSequence/blob/master/randomsequence.h

class PrimeNumberRandomGenerator : public RandomGenerator{
private:
    unsigned int m_index;
    unsigned int m_intermediateOffset;

    static unsigned int permuteQPR(unsigned int x)
    {
        static const unsigned int prime = 23; //4294967291u;
        if (x >= prime)
            return x;  // The 5 integers out of range are mapped to themselves.
        unsigned int residue = ((unsigned long long) pow(x, 2)) % prime;
        return (x <= prime / 2) ? residue : prime - residue;
    }

    unsigned int N_points;
public:
    PrimeNumberRandomGenerator(unsigned int seedBase, unsigned int seedOffset, unsigned int N_points)  {
        m_index = permuteQPR(permuteQPR(seedBase) + 0x682f0161);
        m_intermediateOffset = permuteQPR(permuteQPR(seedOffset) + 0x46790905);
        this->N_points = N_points;
    }

    int getRandomNumber () override {
        return permuteQPR((permuteQPR(m_index++) + m_intermediateOffset) ^ 0x5bf03635);
    }

    void resetGenerator (int min_range, int max_range) override {

    }

    void generateUniqueRandomSample (int * sample) override {

    }


};

#endif //USAC_PRIMENUMBERRANDOMGENERATOR_H