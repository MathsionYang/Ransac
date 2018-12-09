#include "Tests.h"

void Tests::test (cv::Mat points,
                   Model * model,
                   const std::string &img_name,
                   bool gt,
                   const cv::Mat& gt_model) {

    NearestNeighbors nn;
    cv::Mat neighbors, neighbors_dists;

//    std::cout << "get neighbors\n";

    // calculate time of nearest neighbor calculating
    auto begin_time = std::chrono::steady_clock::now();
    nn.getNearestNeighbors_nanoflann(points, model->k_nearest_neighbors, neighbors, false, neighbors_dists);
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<float> fs = end_time - begin_time;
    long nn_time = std::chrono::duration_cast<std::chrono::microseconds>(fs).count();

//    std::cout << "got neighbors\n";

    // ------------------ init -------------------------------------
    Quality * quality = new Quality;
    Sampler * sampler;
    Estimator * estimator;
    TerminationCriteria * termination_criteria;

    initSampler(sampler, model, points.rows, points, neighbors);
    initEstimator(estimator, model, points);

    if (model->sampler == SAMPLER::Prosac) {
        // re init termination criteria for prosac
        ProsacTerminationCriteria *  prosac_termination_criteria_ = new ProsacTerminationCriteria;
        prosac_termination_criteria_->initProsacTerminationCriteria
                (((ProsacSampler *) sampler)->getGrowthFunction(), model, points.rows, estimator);

        termination_criteria = prosac_termination_criteria_;
    } else {
        termination_criteria = new StandardTerminationCriteria;
    }
    // -------------- end of initialization -------------------



    Drawing drawing;
    Logging logResult;

    Ransac ransac (model, sampler, termination_criteria, quality, estimator);
    ransac.setNeighbors(neighbors);
    ransac.run(points);

    RansacOutput * ransacOutput = ransac.getRansacOutput();

    std::cout << model->getName() << "\n";
    std::cout << "\ttime: ";
    long time_mcs = ransacOutput->getTimeMicroSeconds();
    if (model->sampler == SAMPLER::Napsac || model->GraphCutLO) {
        time_mcs += nn_time;
    }
    Time * time = new Time;
    splitTime(time, time_mcs);
    std::cout << time;
    std::cout << "\tMain iterations: " << ransacOutput->getNumberOfMainIterations() << "\n";
    std::cout << "\tLO iterations: " << ransacOutput->getLOIters() <<
    " (where " << ransacOutput->getLOInnerIters () << " (inner iters) and " <<
              ransacOutput->getLOIterativeIters() << " (iterative iters) and " << ransacOutput->getGCIters() << " (GC iters))\n";

    std::cout << "\tpoints under threshold: " << ransacOutput->getNumberOfInliers() << "\n";

    std::cout << "Best model = ...\n" << ransacOutput->getModel ()->returnDescriptor() << "\n";

    if (gt) {
        int GT_num_inliers;
        float error = Quality::getErrorGT(estimator, ransacOutput->getModel(), points.rows, gt_model, &GT_num_inliers);
        if (model->estimator == ESTIMATOR::Homography) {
            float error2;
            int GT_num_inliers2;
            error2 = Quality::getErrorGT(estimator, ransacOutput->getModel(), points.rows, gt_model.inv(), &GT_num_inliers2);
            if (GT_num_inliers2 > GT_num_inliers) {
                GT_num_inliers = GT_num_inliers2;
                error = error2;
            }
        }
        std::cout << "Ground Truth nu mber of inliers for same model parametres is " << GT_num_inliers << "\n";
        std::cout << "Error to GT inliers " << error << "\n";
        std::cout << "GT model = ... \n" << gt_model << "\n";
    }

    // save result and compare with last run
    logResult.compare(model, ransacOutput);
    logResult.saveResult(model, ransacOutput);
    std::cout << "-----------------------------------------------------------------------------------------\n";


    if (model->estimator == ESTIMATOR::Homography) {
        drawing.drawHomographies(img_name, points, ransacOutput->getInliers(), ransacOutput->getModel()->returnDescriptor());
    } else
    if (model->estimator == ESTIMATOR::Fundamental) {
        drawing.drawEpipolarLines(img_name, points.colRange(0,2), points.colRange(2,4), ransacOutput->getModel()->returnDescriptor());
    } else
    if (model->estimator == ESTIMATOR::Line2d) {
        drawing.draw(ransacOutput->getInliers(), ransacOutput->getModel(), points, img_name+".png");
    } else
    if (model->estimator == ESTIMATOR::Essential) {

    } else {
        std::cout << "UNKNOWN ESTIMATOR\n";
    }

}
