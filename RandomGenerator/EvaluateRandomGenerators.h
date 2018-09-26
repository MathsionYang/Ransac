#ifndef USAC_EVALUATERANDOMGENERATORS_H
#define USAC_EVALUATERANDOMGENERATORS_H

#include "SimpleRandomGenerator.h"
#include "ArrayRandomGenerator.h"
#include "UniformRandomGenerator.h"
#include "PrimeNumberRandomGenerator.h"
#include "ShuffleRandomGenerator.h"

void calculateEntropy (RandomGenerator * random_generator, int size, std::string name);
void getAverageTime (RandomGenerator * random_generator, int size, int unique_set_size, std::string name);

void evaluateRandomGenerators () {
    RandomGenerator * array_random_generator = new ArrayRandomGenerator;
    RandomGenerator * simple_random_generator = new SimpleRandomGenerator;
    RandomGenerator * uniform_random_generator = new UniformRandomGenerator;
    RandomGenerator * prime_random_generator = new PrimeNumberRandomGenerator;
    RandomGenerator * shuffle_random_generator = new ShuffleRandomGenerator;

    int size = 10000;
    int uniques_set_size = 200;
    calculateEntropy (array_random_generator, size, "array");
    getAverageTime (array_random_generator, size, uniques_set_size, "array");

    calculateEntropy (simple_random_generator, size, "simple");
    getAverageTime (simple_random_generator, size, uniques_set_size, "simple");

    calculateEntropy (uniform_random_generator, size, "uniform");
    getAverageTime (uniform_random_generator, size, uniques_set_size, "uniform");

    calculateEntropy (prime_random_generator, size, "prime");
    getAverageTime (prime_random_generator, size, uniques_set_size, "prime");

    calculateEntropy (shuffle_random_generator, size, "shuffle");
    getAverageTime (shuffle_random_generator, size, uniques_set_size, "shuffle");
}

void calculateEntropy (RandomGenerator * random_generator, int size, std::string name) {
    std::cout << name << " random generator, size =  " << size << "\n";
    random_generator->resetGenerator(0, size);

    std::vector<int> histogram (size, 0);

    for (int i = 0; i < size; i++) {
        histogram[random_generator->getRandomNumber()]++;
    }

    float E = 0;
    for (int i = 0; i < size; i++) {
//        std::cout << histogram[i] << " " ;
        if (histogram[i] == 0) continue;
        E -= ((float)histogram[i]/size) * log ((float) histogram[i]/size);
    }
//    std::cout << '\n';

    std::cout << "Entropy (min: 0 (non uniform), max " << (-log ((float) 1/size)) << " (full uniform)): " << E << '\n';
//    std::cout << "------------------------------------------------------------\n";
}

void getAverageTime (RandomGenerator * random_generator, int size, int unique_set_size, std::string name) {
//    std::cout << name << " random generator, size =  " << size << " unique_set_size = " << unique_set_size << "\n";

    random_generator->resetGenerator(0, size);

    auto begin_time = std::chrono::steady_clock::now();

    for (int i = 0; i < size; i++) {
//        std::cout << random_generator->getRandomNumber() << " ";
        random_generator->getRandomNumber();
    }
//    std::cout << '\n';

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<float> fs = end_time - begin_time;
    std::chrono::microseconds t = std::chrono::duration_cast<std::chrono::microseconds>(fs);
    std::cout << "getRandomNumber time = " << t.count() << '\n';


    int * sample = new int [unique_set_size];

    begin_time = std::chrono::steady_clock::now();
    for (int i = 0; i < size; i++) {
        random_generator->generateUniqueRandomSet(sample, unique_set_size);
    }
    end_time = std::chrono::steady_clock::now();
    fs = end_time - begin_time;
    t = std::chrono::duration_cast<std::chrono::microseconds>(fs);
    std::cout << "generateUniqueRandomSet time = " << t.count() << '\n';
    std::cout << "------------------------------------------------------------\n";
}

#endif //USAC_EVALUATERANDOMGENERATORS_H