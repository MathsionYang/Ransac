#include "Tests.h"
#include "../usac/sampler/uniform_sampler.hpp"
#include "../usac/sampler/prosac_sampler.hpp"
#include "../usac/sampler/evsac_sampler.hpp"
#include "../usac/sampler/napsac_sampler.hpp"
#include "../usac/sampler/progressive_sampler.hpp"

void Tests::initProsac (Sampler *& sampler, unsigned int sample_number, unsigned int points_size) {
    sampler = new ProsacSampler;
    ProsacSampler * prosac_sampler = (ProsacSampler *)sampler;
    prosac_sampler->initProsacSampler (sample_number, points_size);
}

void Tests::initUniform (Sampler *& sampler, unsigned int sample_number, unsigned int points_size, bool reset_time) {
    sampler = new UniformSampler (reset_time);
    sampler->setSampleSize(sample_number);
    sampler->setPointsSize(points_size);
}

void Tests::initNapsac (Sampler *& sampler, const cv::Mat &neighbors, const std::vector<std::vector<int>> &ns, Model * model) {
    int points_size = std::max ((int) neighbors.rows, (int) ns.size());

    sampler = new NapsacSampler(model, points_size, model->reset_random_generator);
    if (model->neighborsType == NeighborsSearch::Nanoflann) {
        assert(! neighbors.empty());
        ((NapsacSampler *) sampler)->setNeighbors(neighbors);
    } else {
        assert(! ns.empty());
        ((NapsacSampler *) sampler)->setNeighbors(ns);
    }

}

void Tests::initEvsac (Sampler *& sampler, cv::InputArray points, unsigned int sample_number,
                       unsigned int points_size, unsigned int k_nearest_neighbors) {

    sampler = new EvsacSampler(points, points_size, k_nearest_neighbors, sample_number);
}

void Tests::initGraduallyIncreasingSampler (Sampler *& sampler, cv::InputArray points, unsigned int sample_number) {
    sampler = new ProgressiveNapsac(points, sample_number);
}

void Tests::initProsacNapsac1 (Sampler *& sampler, Model * model, const cv::Mat &nearest_neighors) {

}

void Tests::initProsacNapsac2 (Sampler *& sampler, Model * model, const cv::Mat &nearest_neighors) {

}

void Tests::initSampler (Sampler *& sampler, Model * model, unsigned int points_size, cv::InputArray points, const cv::Mat& neighbors, std::vector<std::vector<int>> ns) {
    Tests tests;
    if (model->sampler == SAMPLER::Uniform) {
        tests.initUniform(sampler, model->sample_size, points_size, model->reset_random_generator);
    } else if (model->sampler == SAMPLER::Prosac) {
        tests.initProsac(sampler, model->sample_size, points_size);
    } else if (model->sampler == SAMPLER::Napsac) {
        assert(model->k_nearest_neighbors > 0);
        tests.initNapsac(sampler, neighbors, ns, model);
    } else if (model->sampler == SAMPLER::Evsac) {
        assert(model->k_nearest_neighbors > 0);
        tests.initEvsac(sampler, points, model->sample_size, points_size, model->k_nearest_neighbors);
    } else if (model->sampler == SAMPLER::ProgressiveNAPSAC) {
        tests.initGraduallyIncreasingSampler(sampler, points, model->sample_size);
    } else {
        std::cout << "UNKOWN SAMPLER\n";
        exit (100);
    }
}