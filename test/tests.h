#ifndef TESTS_TESTS_H
#define TESTS_TESTS_H

#include "test_precomp.hpp"

#include "../usac/estimator/estimator.hpp"
#include "../usac/quality/quality.hpp"
#include "../usac/ransac/ransac.hpp"
#include "statistical_results.h"
#include "../usac/sampler/prosac_sampler.hpp"
#include "../usac/termination_criteria/prosac_termination_criteria.hpp"
#include "../usac/utils/nearest_neighbors.hpp"
#include "../detector/detector.h"

#include "../detector/Reader.h"
#include "../dataset/GetImage.h"
#include "../helper/drawing/Drawing.h"

class Tests {
public:
    static void testLineFitting ();
    static void testHomographyFitting ();
    static void testFundamentalFitting ();
    static void testEssentialFitting ();
    static void testAffineFitting ();
    
    static void testNeighborsSearchCell ();
    static void testNeighborsSearch ();
    static void testFindMedian ();
    static void testFindNearestNeighbors (int knn=7);
    static void testInv ();
    static void testDensityProsac ();
    static void testDensityOptimalKnn();
    
    static void test (cv::Mat points,
                  Model * model,
                  const std::string &img_name,
                  DATASET dataset,
                  bool gt,
                  const std::vector<int>& gt_inliers);


    static void detectAndSaveFeatures (const std::vector<std::string>& dataset) {
//        std::string folder = "../dataset/homography/";
        std::string folder = "../dataset/Lebeda/kusvod2/";
//        std::string folder = "../dataset/adelaidermf/";

        for (const std::string &name : dataset) {
            std::cout << name << "\n";
            cv::Mat points;

            cv::Mat image1 = cv::imread (folder+name+"A.png");
            cv::Mat image2 = cv::imread (folder+name+"B.png");

            if (image1.empty()) {
                image1 = cv::imread (folder+name+"A.jpg");
                image2 = cv::imread (folder+name+"B.jpg");
            }

            DetectFeatures(folder+"sift_update/"+name+"_pts.txt", image1, image2, points);
        }
    }

    static std::string sampler2string (SAMPLER sampler) {
        if (sampler == SAMPLER::Prosac) return "prosac";
        if (sampler == SAMPLER::Uniform) return "uniform";
        if (sampler == SAMPLER::Napsac) return "napsac";
        if (sampler == SAMPLER::Evsac) return "evsac";
        if (sampler == SAMPLER::ProgressiveNAPSAC) return "pronapsac";
        return "";
    }

    static std::string estimator2string (ESTIMATOR estimator) {
        if (estimator == ESTIMATOR::Line2d) return "line2d";
        if (estimator == ESTIMATOR::Homography) return "homography";
        if (estimator == ESTIMATOR::Fundamental) return "fundamental";
        if (estimator == ESTIMATOR::Essential) return "essential";
        return "";
    }

    static std::string nearestNeighbors2string (NeighborsSearch nn) {
        if (nn == NeighborsSearch::Grid) return "grid";
        if (nn == NeighborsSearch::Nanoflann) return "nanoflann";
        return "";
    }

    static std::string getComputerInfo () {
        return "";
    }
    /*
     * Display average results such as computational time,
     * number of inliers of N runs of Ransac.
     */
    static void getStatisticalResults (const cv::Mat& points,
                                Model * const model,
                                int N,
                                bool GT,
                                const std::vector<int>& gt_inliers, bool get_results,
                                StatisticalResults * statistical_results) {

//        std::cout << "Testing " << estimator2string(model->estimator) << "\n";
//        std::cout << "with " << sampler2string(model->sampler) << " sampler\n";
//        std::cout << "with " << nearestNeighbors2string(model->neighborsType) << " neighbors searching\n";
//        std::cout << "with cell size = " << model->cell_size << "\n";
//        std::cout << "LO " << model->LO << "\n";
//        std::cout << "GC " << model->GraphCutLO << "\n";
//        std::cout << "SPRT " << model->Sprt << "\n";
//        std::cout << N << " times \n";
        
        Estimator * estimator;
        initEstimator(estimator, model->estimator, points);

        std::vector<long> times(N);
        std::vector<float> num_inlierss(N);
        std::vector<float> num_iterss(N);
        std::vector<float> num_lo_iterss(N);
        std::vector<float> errorss(N);
        
        long time = 0;
        float num_inliers = 0;
        float errors = 0;
        float num_iters = 0;
        float num_lo_iters = 0;

        int fails_10 = 0;
        int fails_25 = 0;
        int fails_50 = 0;

        cv::Mat neighbors, neighbors_dists;
        std::vector<std::vector<int>> neighbors_v;

        // calculate time of nearest neighbor calculating
        auto begin_time = std::chrono::steady_clock::now();
        if (model->sampler == SAMPLER::Napsac || model->lo == LocOpt::GC) {
            if (model->neighborsType == NeighborsSearch::Nanoflann) {
                NearestNeighbors::getNearestNeighbors_nanoflann(points, model->k_nearest_neighbors, neighbors, false,
                                                                neighbors_dists);
            } else {
                NearestNeighbors::getGridNearestNeighbors(points, model->cell_size, neighbors_v);
            }
        }
        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<float> fs = end_time - begin_time;
        long nn_time = std::chrono::duration_cast<std::chrono::microseconds>(fs).count();

        // Calculate Average of number of inliers, number of iteration,
        // time and average error.
        // If we have GT number of inliers, then find number of fails model.
        for (int i = 0; i < N; i++) {
//            std::cout << "i = " << i << "\n";
            Ransac ransac (model, points);
            ransac.run();
            RansacOutput ransacOutput = *ransac.getRansacOutput();

            if (model->sampler == SAMPLER::Napsac || model->lo == LocOpt::GC) {
                times[i] = ransacOutput.getTimeMicroSeconds() + nn_time;
            } else {
                times[i] = ransacOutput.getTimeMicroSeconds();
            }

            num_inlierss[i] = ransacOutput.getNumberOfInliers();
            num_iterss[i] = ransacOutput.getNumberOfMainIterations();
            num_lo_iterss[i] = ransacOutput.getNUmberOfLOIterarations();

            time += times[i];
            num_inliers += ransacOutput.getNumberOfInliers();
            num_iters += ransacOutput.getNumberOfMainIterations();
            num_lo_iters = ransacOutput.getNUmberOfLOIterarations();

            if (GT) {
                float error = Quality::getErrorToGTInliers(estimator, ransacOutput.getModel()->returnDescriptor(), gt_inliers);
                errors += error;
                errorss[i] = error;

                /*
                 * If ratio of number of inliers and number of
                 * Ground Truth inliers is less than 50% then
                 * it is fail.
                 */
                std::vector<int> est_inliers = ransacOutput.getInliers();
                float matches = 0;
                for (int inl = 0; inl < gt_inliers.size(); inl++) {
                    for (int j = 0; j < num_inlierss[i]; j++) {
                        if (gt_inliers[inl] == est_inliers[j]) {
                            matches++;
                            break;
                        }
                    }
                }
                if (matches / gt_inliers.size() < 0.10) {
                    fails_10++;
                    fails_25++;
                    fails_50++;
                } else if (matches / gt_inliers.size() < 0.25) {
                    fails_25++;
                    fails_50++;
                } else if (matches / gt_inliers.size() < 0.50) {
                    fails_50++;
                }
//            std::cout << "----------------------------------------------------------------\n";
            }
        }
        
        StatisticalResults results;
        if (GT) {
            results.num_fails_10 = fails_10;
            results.num_fails_25 = fails_25;
            results.num_fails_50 = fails_50;
            results.avg_avg_error = errors/N;
        }

        results.avg_time_mcs = time/N;
        results.avg_num_inliers = num_inliers/N;
        results.avg_num_iters = num_iters/N;
        results.avg_num_lo_iters = num_lo_iters/N;


        long time_ = 0; float iters_ = 0, lo_iters_ = 0, inl_ = 0, err_ = 0;
        // Calculate sum ((xj - x)^2)
        for (int j = 0; j < N; j++) {
            time_ += pow (results.avg_time_mcs - times[j], 2);
            inl_ += pow (results.avg_num_inliers - num_inlierss[j], 2);
            err_ += pow (results.avg_avg_error - errorss[j], 2);
            iters_ += pow (results.avg_num_iters - num_iterss[j], 2);
            lo_iters_ += pow (results.avg_num_lo_iters - num_lo_iterss[j], 2);
        }

        // Calculate standart deviation
        int biased = 1;
        results.std_dev_time_mcs = sqrt (time_/(N-biased));
        results.std_dev_num_inliers = sqrt (inl_/(N-biased));
        results.std_dev_num_iters = sqrt (iters_/(N-biased));
        results.std_dev_num_lo_iters = sqrt (lo_iters_/(N-biased));

        if (GT) {
            results.std_dev_avg_error = sqrt (err_/(N-biased));
        }

        // Sort results for median
        std::sort (times.begin(), times.end(), [] (long a, long b) { return a < b; });
        std::sort (num_inlierss.begin(), num_inlierss.end(), [] (float a, float b) { return a < b; });
        std::sort (num_iterss.begin(), num_iterss.end(), [] (float a, float b) { return a < b; });
        std::sort (num_lo_iterss.begin(), num_lo_iterss.end(), [] (float a, float b) { return a < b; });
        std::sort (errorss.begin(), errorss.end(), [] (float a, float b) { return a < b; });

        if (GT) {
            results.worst_case_error = errorss[N-1];
        }
        results.worst_case_num_inliers = num_inlierss[0];

        // Calculate median of results for N is even
        results.median_time_mcs = (times[N/2-1] + times[N/2])/2;
        results.median_num_inliers = (num_inlierss[N/2-1] + num_inlierss[N/2])/2;
        results.median_num_iters = (num_iterss[N/2-1] + num_iterss[N/2])/2;
        results.median_num_lo_iters = (num_lo_iterss[N/2-1] + num_lo_iterss[N/2])/2;
        if (GT) {
            results.median_avg_error = (errorss[N/2-1] + errorss[N/2])/2;
        }

//       std::cout << &results << "\n";

        if (get_results) {
            statistical_results->copyFrom(&results);
        }
        delete (estimator);
    }

    static void testOpenCV (const cv::Mat &points, Model * model, const std::vector<int> &gt_inliers,
                            int N_runs, float *avg_err, float * avg_time) {
        cv::Mat_<float> m;
        Estimator * estimator;
        initEstimator(estimator, model->estimator, points);

        float errors = 0, time = 0;
        for (unsigned int run = 0; run < N_runs; run++) {
            auto begin_time = std::chrono::steady_clock::now();

            if (model->estimator == ESTIMATOR::Homography) {
                m = cv::findHomography(points.colRange(0,2), points.colRange(2,4), cv::RANSAC,
                                       model->threshold, cv::noArray(), model->max_iterations, model->confidence);
            } else if (model->estimator == ESTIMATOR::Essential) {
                m = cv::findEssentialMat(points.colRange(0,2), points.colRange(2,4),
                                         1.0 /*focal*/, cv::Point2d(0, 0) /*pp*/,cv::RANSAC,
                                         model->confidence, model->threshold);
            } else if (model->estimator == ESTIMATOR::Fundamental) {
                m = cv::findFundamentalMat(points.colRange(0,2), points.colRange(2,4), cv::RANSAC,
                                       model->threshold, model->confidence);
            } else {
                std::cout << "opencv undefined estimator\n";
                exit(111);
            }
            std::chrono::duration<float> fs = std::chrono::steady_clock::now() - begin_time;
//            std::cout << Quality::getErrorToGTInliers(estimator, m, gt_inliers) << " = err\n";
            errors += Quality::getErrorToGTInliers(estimator, m, gt_inliers); // not inverse!
            time += std::chrono::duration_cast<std::chrono::microseconds>(fs).count();
        }

        *avg_err = errors / N_runs;
        *avg_time = time / N_runs;
        delete (estimator);
    }

    //todo add functions for storeResults () and showResults


//    void storeResults () {
//
//    }

    static void run_all_tests ();
};

#endif //TESTS_TESTS_H