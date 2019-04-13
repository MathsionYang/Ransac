#include "tests.h"

void Tests::testDensityProsac () {
	DATASET dataset = DATASET::Homogr_SIFT;
	std::vector<std::string> points_filename = Dataset::getDataset(dataset);

    int num_images = points_filename.size();

    std::vector<cv::Mat_<float>> sift_sorted_points_imgs;
    std::vector<cv::Mat_<float>> dense_sorted_points_imgs;
    
    std::vector<std::vector<int>> sift_sorted_gt_inliers;
    std::vector<std::vector<int>> dense_sorted_gt_inliers;
    
    std::vector<long> dense_sorting_time;

    int N_runs = 200;
    int knn = 5;
    float threshold = 2;
    float confidence = 0.95;
    float cell_size = 50;

    for (const std::string &img_name : points_filename) {
        std::cout << "get points for " << img_name << "\n";
        
        ImageData gt_data (dataset, img_name);
        cv::Mat points = gt_data.getPoints();
        cv::Mat sift_sorted_points = gt_data.getSortedPoints();
        cv::Mat dense_sorted_points;
        
        std::vector<int> gt_inliers_ = gt_data.getGTInliers(threshold);
        std::vector<int> dense_inliers;

        // auto begin_time = std::chrono::steady_clock::now();
        densitySort (points, 10, dense_sorted_points, gt_inliers_, dense_inliers);
        // std::chrono::duration<float> fs = std::chrono::steady_clock::now() - begin_time;
        // dense_sorting_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(fs).count());

        sift_sorted_points_imgs.emplace_back (sift_sorted_points);
        dense_sorted_points_imgs.emplace_back (dense_sorted_points);

        sift_sorted_gt_inliers.push_back(gt_data.getGTInliersSorted(threshold));
        dense_sorted_gt_inliers.push_back(dense_inliers);

        std::cout << "inliers size " << gt_inliers_.size() << "\n";
    }

    std::cout << N_runs <<" for each " << num_images  << "\n";

    Model model (threshold, confidence, knn, ESTIMATOR::Homography, SAMPLER::Prosac);
    // Model model (threshold, confidence, knn, ESTIMATOR::Fundamental, SAMPLER::Prosac);
    model.lo = LocOpt ::NullLO;
    model.setNeighborsType(NeighborsSearch::Grid);
    model.setCellSize(cell_size);
    model.ResetRandomGenerator(true);

    int img = 0;
    float avg_avg_avg_error;
    float avg_avg_time;

    avg_avg_avg_error = 0; avg_avg_time = 0;
    StatisticalResults statistical_results;
    for (const std::string &img_name : points_filename) {
            Tests::getStatisticalResults(sift_sorted_points_imgs[img], &model, N_runs,
                                        true, sift_sorted_gt_inliers[img], true, &statistical_results);

        float avg_avg_error = statistical_results.avg_avg_error;
        float avg_time = statistical_results.avg_time_mcs;
    	std::cout << avg_time << " " << avg_avg_error << "\n";    
        avg_avg_avg_error += avg_avg_error;
        avg_avg_time += avg_time;

        img++;
    }
    std::cout << "SIFT Average avg. time & Average avg. avg. error:\n";
    std::cout << (avg_avg_time / num_images) << " " << (avg_avg_avg_error / num_images) << "\n";

    img = 0;
    avg_avg_avg_error = 0; avg_avg_time = 0;
    for (const std::string &img_name : points_filename) {
            Tests::getStatisticalResults(dense_sorted_points_imgs[img], &model, N_runs,
                                        true, dense_sorted_gt_inliers[img], true, &statistical_results);

        float avg_avg_error = statistical_results.avg_avg_error;
        float avg_time = statistical_results.avg_time_mcs;
    	std::cout << avg_time << " " << avg_avg_error << "\n";    
        avg_avg_avg_error += avg_avg_error;
        avg_avg_time += avg_time;

        img++;
    }
    std::cout << "DENSE Average avg. time & Average avg. avg. error:\n";
    std::cout << (avg_avg_time / num_images) << " " << (avg_avg_avg_error / num_images) << "\n";
}

void Tests::testDensityOptimalKnn () {
    DATASET dataset = DATASET::Homogr_SIFT;
    std::vector<std::string> points_filename = Dataset::getDataset(dataset);

    int num_images = points_filename.size();

    std::vector<cv::Mat> points_imgs;
    std::vector<std::vector<int>> inliers_imgs;
    
    int N_runs = 100;
    int knn = 5;
    float threshold = 2;
    float confidence = 0.95;

    for (const std::string &img_name : points_filename) {
        std::cout << "get points for " << img_name << "\n";
        ImageData gt_data (dataset, img_name);
        cv::Mat points = gt_data.getPoints();

        points_imgs.emplace_back(points);

        std::vector<int> gt_inliers_ = gt_data.getGTInliers(threshold);
        inliers_imgs.emplace_back(gt_inliers_);

        std::cout << "inliers size " << gt_inliers_.size() << "\n";
        std::cout << "points size " << points.rows << "\n";
    }

    Model model (threshold, confidence, knn, ESTIMATOR::Homography, SAMPLER::Prosac);
    // Model model (threshold, confidence, knn, ESTIMATOR::Fundamental, SAMPLER::Prosac);
    model.lo = LocOpt ::NullLO;

    StatisticalResults statistical_results;

    for (int k = 9; k < 16; k++) {

        float avg_avg_avg_error = 0, avg_avg_time = 0;
        int img = 0;
        for (const std::string &img_name : points_filename) {
            std::cout << img_name << "\n";
            cv::Mat dense_sorted_points;
            std::vector<int> dense_inliers;

            auto begin_time = std::chrono::steady_clock::now();
            densitySort (points_imgs[img], k, dense_sorted_points, inliers_imgs[img], dense_inliers);
            std::chrono::duration<float> fs = std::chrono::steady_clock::now() - begin_time;
            float dense_sorting_time = std::chrono::duration_cast<std::chrono::microseconds>(fs).count();

            Tests::getStatisticalResults(dense_sorted_points, &model, N_runs,
                                        true, dense_inliers, true, &statistical_results);

//            std::cout << statistical_results.avg_time_mcs << " " << statistical_results.avg_avg_error << "\n";
            avg_avg_avg_error += statistical_results.avg_avg_error;
            avg_avg_time += statistical_results.avg_time_mcs + dense_sorting_time;

            img++;
        }
        std::cout << "k = "<< k <<" Average avg. time & Average avg. avg. error:\n";
        std::cout << (avg_avg_time / num_images) << " " << (avg_avg_avg_error / num_images) << "\n";
    }
}