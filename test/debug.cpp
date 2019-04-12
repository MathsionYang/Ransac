#include "tests.h"

void run_test (Model model, const cv::Mat &points, int num_runs_per_test=5) {
    for (int i = 0; i < num_runs_per_test; i++) {
        Ransac ransac (&model, points);
        ransac.run();
    }
}

void tests_by_estimator (ESTIMATOR estimator, const cv::Mat &points, const cv::Mat &sorted_points) {
    if (1) {
        //  UNIFORM
        Model model(2, 0.95, 8, estimator, SAMPLER::Uniform);
        std::cout << "test "+ Tests::estimator2string(estimator) + " uniform\n";
        run_test(model, points);
    }

    if (1) {
        // UNIFORM + SPRT
        Model model(2, 0.95, 8, estimator, SAMPLER::Uniform);
        model.setSprt(true);
        std::cout << "test "+ Tests::estimator2string(estimator) + " uniform + sprt\n";
        run_test(model, points);
    }

    if (1) {
        // PROSAC
        Model model(2, 0.95, 8, estimator, SAMPLER::Prosac);
        std::cout << "test "+ Tests::estimator2string(estimator) + " prosac\n";
        run_test(model, sorted_points);
    }

    if (1) {
        // GC + GRID
        Model model(2, 0.95, 8, estimator, SAMPLER::Uniform);
        model.lo = LocOpt ::GC;
        model.setNeighborsType(NeighborsSearch::Grid);
        std::cout << "test "+ Tests::estimator2string(estimator) + " GC grid\n";
        run_test(model, points);
    }

    if (1) {
        // GC + NANOFLANN
        Model model(2, 0.95, 8, estimator, SAMPLER::Uniform);
        model.lo = LocOpt ::GC;
        model.setNeighborsType(NeighborsSearch::Nanoflann);
        std::cout << "test "+ Tests::estimator2string(estimator) + " GC nanoflann\n";
        run_test(model, points);
    }

    if (1) {
        // Locally optimized
        Model model(2, 0.95, 8, estimator, SAMPLER::Uniform);
        model.lo = LocOpt ::InItLORsc;
        std::cout << "test "+ Tests::estimator2string(estimator) + " locally optimized\n";
        run_test(model, points);
    }

    if (1) {
        // Fixing Locally optimized
        Model model(2, 0.95, 8, estimator, SAMPLER::Uniform);
        model.lo = LocOpt ::InItFLORsc;
        std::cout << "test "+ Tests::estimator2string(estimator) + " fixing locally optimized\n";
        run_test(model, points);
    }

    if (1) {
        // Fixing Locally optimized
        Model model(2, 0.95, 8, estimator, SAMPLER::Napsac);
        model.setNeighborsType(NeighborsSearch::Grid);
        std::cout << "test "+ Tests::estimator2string(estimator) + " napsac grid\n";
        run_test(model, points);
    }

    if (1) {
        // Fixing Locally optimized
        Model model(2, 0.95, 8, estimator, SAMPLER::Napsac);
        model.setNeighborsType(NeighborsSearch::Nanoflann);
        std::cout << "test "+ Tests::estimator2string(estimator) + " napsac nanoflann\n";
        run_test(model, points);
    }
}
void Tests::run_all_tests() {
    DATASET dataset = DATASET::Homogr_SIFT;
    std::string img_name = "graf";
    ImageData gt_data (dataset, img_name);
    cv::Mat points = gt_data.getPoints();
    cv::Mat sorted_points = gt_data.getSortedPoints();

    if (true) {
//        HOMOGRAPHY
        tests_by_estimator(ESTIMATOR::Homography, points, sorted_points);
    }
    if (true) {
//        FUNDAMENTAL
        tests_by_estimator(ESTIMATOR::Fundamental, points, sorted_points);
    }
    if (true) {
//        ESSENTIAL
        tests_by_estimator(ESTIMATOR::Essential, points, sorted_points);
    }
    if (true) {
//        AFFINE
        tests_by_estimator(ESTIMATOR::Affine, points, sorted_points);
    }

}
