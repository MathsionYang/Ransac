#include "tests.h"
#include "../helper/Logging.h"

void Tests::test (cv::Mat points,
                   Model * model,
                   const std::string &img_name,
                   DATASET dataset,
                   bool gt,
                   const std::vector<int>& gt_inliers) {

    cv::Mat neighbors, neighbors_dists;

//    std::cout << "get neighbors\n";
    std::vector<std::vector<int>> neighbors_v;

    long nn_time = 0;
    if (model->sampler == SAMPLER::Napsac || model->lo == LocOpt::GC) {
        // calculate time of nearest neighbor calculating
        auto begin_time = std::chrono::steady_clock::now();
        if (model->neighborsType == NeighborsSearch::Nanoflann) {
            NearestNeighbors::getNearestNeighbors_nanoflann(points, model->k_nearest_neighbors, neighbors, false, neighbors_dists);
        } else {
            NearestNeighbors::getGridNearestNeighbors(points, model->cell_size, neighbors_v);
        }
        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<float> fs = end_time - begin_time;
        nn_time = std::chrono::duration_cast<std::chrono::microseconds>(fs).count();
    }

    Ransac ransac (model, points);

    ransac.run();

    RansacOutput * ransacOutput = ransac.getRansacOutput();

    std::cout << Tests::sampler2string(model->sampler) +"_"+Tests::estimator2string(model->estimator) << "\n";
    std::cout << "\ttime: ";
    long time_mcs = ransacOutput->getTimeMicroSeconds();
    if (model->sampler == SAMPLER::Napsac || model->lo == LocOpt::GC) {
        time_mcs += nn_time;
    }
    Time * time = new Time;
    splitTime(time, time_mcs);
    std::cout << time;
    std::cout << "\tMain iterations: " << ransacOutput->getNumberOfMainIterations() << "\n";
    std::cout << "\tLO iterations: " << ransacOutput->getNUmberOfLOIterarations() << "\n";

    std::cout << "\tpoints under threshold: " << ransacOutput->getNumberOfInliers() << "\n";

    std::cout << "Best model = ...\n" << ransacOutput->getModel ()->returnDescriptor() << "\n";

    if (gt) {
        Estimator * estimator;
        initEstimator(estimator, model->estimator, points);
        float error = Quality::getErrorToGTInliers(estimator, ransacOutput->getModel()->returnDescriptor(), gt_inliers);
        std::cout << "Ground Truth number of inliers for same model parametres is " << gt_inliers.size() << "\n";
        std::cout << "Error to GT inliers " << error << "\n";
    }

    // save result and compare with last run
    Logging::compare(model, ransacOutput);
    Logging::saveResult(model, ransacOutput);
    std::cout << "-----------------------------------------------------------------------------------------\n";

    Drawing::draw(ransacOutput->getModel(), dataset, img_name);

    cv::Mat_<float> m = cv::findHomography(points.colRange(0,2), points.colRange(2,4), cv::RANSAC,
                                   model->threshold, cv::noArray(), model->max_iterations, model->confidence);
    std::cout << m << "=m\n\n";
    Estimator * estimator1 = new HomographyEstimator(points);
    std::cout << Quality::getErrorToGTInliers(estimator1, m, gt_inliers) << " err\n";
//    Model * model1 = new Model(model);
//    model1->setDescriptor(m);
//    Model * model2 = new Model(model);
//    model2->setDescriptor(m.inv());
//
//    Drawing::draw(model1, dataset, img_name);
//    Drawing::draw(model2, dataset, img_name);

    delete (ransacOutput);
}
