#include "tests.h"

void Tests::testAffineFitting () {
	float threshold = 2;

    DATASET dataset = DATASET::Homogr_SIFT;
    std::string img_name = "graf";

    ImageData gt_data (dataset, img_name);

    cv::Mat points = gt_data.getPoints();
    cv::Mat sorted_points = gt_data.getSortedPoints();

    std::vector<int> gt_inliers = gt_data.getGTInliers(threshold);
    std::vector<int> gt_sorted_inliers = gt_data.getGTInliersSorted(threshold);

    unsigned int points_size = (unsigned int) points.rows;
    std::cout << "points size " << points_size << "\n";
    std::cout << "sorted points size " << sorted_points.rows << "\n";

    unsigned int knn = 7;
    float confidence = 0.95;

    std::cout << "gt inliers " << gt_inliers.size() << "\n";
    std::cout << "gt inliers sorted " << gt_sorted_inliers.size() << "\n";

//     ---------------------- uniform ----------------------------------
   	Model model (threshold, confidence, knn, ESTIMATOR::Affine, SAMPLER::Uniform);
//     --------------------------------------------------------------

//     ---------------------- napsac ----------------------------------
//    Model model (threshold, confidence, knn, ESTIMATOR::Affine, SAMPLER::Napsac);
//     --------------------------------------------------------------

// ------------------ prosac ---------------------
//    Model model (threshold, confidence, knn, ESTIMATOR::Affine, SAMPLER::Prosac);
//     -------------------------------------------------


     model.lo = LocOpt ::NullLO;
     model.setSprt(0);
     model.setCellSize(50);
     model.setNeighborsType(NeighborsSearch::Grid);
     model.ResetRandomGenerator(false);

    if (model.sampler ==  SAMPLER::Prosac) {
       test (sorted_points, &model, img_name, dataset, true, gt_sorted_inliers);
        // getStatisticalResults(sorted_points, &model, 500, true, gt_sorted_inliers, false, nullptr);
    } else {
       test (points, &model, img_name, dataset, true, gt_inliers);
//        getStatisticalResults(points, &model, 500, true, gt_inliers, false, nullptr);
    }
}