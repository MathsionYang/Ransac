#include "Tests.h"

#include <cstdio>
#include <iostream>
#include <chrono>
#include <vector>
#include <opencv2/core/types.hpp>

#include "../Detector/ReadPoints.h"

#include "../Usac/DLT/DLT.h"
#include "../Usac/DLT/GetNormalizingTransformation.h"
#include "../Usac/DLT/NormalizedDLT.h"
#include "../Usac/Model.h"
#include "../Usac/Sampler/Sampler.h"
#include "../Usac/Ransac/Ransac.h"
#include "../Usac/Sampler/UniformSampler.h"
#include "../Usac/Estimator/HomographyEstimator.h"
#include "../Usac/Helper/Drawing/Drawing.h"
#include "../Usac/Helper/Logging.h"

#include "../Generator/generator.h"

void testHomography (cv::InputArray points, Model * const model, Sampler * const sampler, std::vector<std::string> images_filename, std::string points_filename);
void storeResults ();

void Tests::testHomographyFitting() {
    std::string img_name = "graf";
    std::string points_filename = "../dataset/homography/"+img_name+"_pts.txt";
    std::vector<std::string> images_filename;
    images_filename.push_back("../dataset/homography/"+img_name+"A.png");
    images_filename.push_back("../dataset/homography/"+img_name+"B.png");

    cv::Mat points, points1, points2;
    read_points (points1, points2, points_filename);
    cv::hconcat(points1, points2, points);

    Model *homography_model = new Model (3, 4, 0.99, 0, "homography");
    Sampler *uniform_sampler = new UniformSampler;
    uniform_sampler->setSampleSize(homography_model->sample_number);
    uniform_sampler->setPointsSize(points1.rows);

    testHomography (points, homography_model, uniform_sampler, images_filename, points_filename);

    Estimator * homograpy_estimator = new HomographyEstimator (points);
    TerminationCriteria *termination_criteria = new TerminationCriteria;
    Quality *quality = new Quality;

//    runNTimes(points, homograpy_estimator, homography_model, uniform_sampler, termination_criteria, quality, 1000);

    storeResults();
}

void testHomography (cv::InputArray points, Model * const model, Sampler * const sampler, std::vector<std::string> images_filename, std::string points_filename) {
    Estimator * homograpy_estimator = new HomographyEstimator (points);
    Drawing drawing;
    Logging logResult;
    TerminationCriteria *termination_criteria = new TerminationCriteria;
    Quality *quality = new Quality;

    Ransac ransac (*model, *sampler, *termination_criteria, *quality);
    ransac.run(points, homograpy_estimator);
    std::cout << model->model_name << " time: ";
    ransac.getQuality()->printTime();
    std::cout << model->model_name << " iterations: " << ransac.getQuality()->getIterations() << "\n";
    std::cout << model->model_name << " points under threshold: " << ransac.getQuality()->getNumberOfPointsUnderThreshold() << "\n";

    // save result and compare with last run
    logResult.compare(model, ransac.getQuality());
    logResult.saveResult(model, ransac.getQuality());
    std::cout << "-----------------------------------------------------------------------------------------\n";

    drawing.drawHomographies(images_filename, points_filename, ransac.most_inliers, ransac.best_model.returnDescriptor());
}

void storeResults () {
    std::vector<std::string> points_filename = {"adam_pts.txt", "Brussels_pts.txt", "LePoint1_pts.txt",
                                                "boat_pts.txt",          "CapitalRegion_pts.txt",  "LePoint2_pts.txt",
                                                "BostonLib_pts.txt",     "city_pts.txt",           "LePoint3_pts.txt",
                                                "Boston_pts.txt",        "Eiffel_pts.txt",         "WhiteBoard_pts.txt",
                                                "BruggeSquare_pts.txt",  "ExtremeZoom_pts.txt",
                                                "BruggeTower_pts.txt",   "graf_pts.txt"};
    TerminationCriteria *termination_criteria = new TerminationCriteria;
    Quality *quality = new Quality;

    for (std::string img_name : points_filename) {
        cv::Mat points1, points2;
        read_points (points1, points2, "../dataset/homography/"+img_name);
        cv::hconcat(points1, points2, points1);

        Estimator * homograpy_estimator = new HomographyEstimator (points1);
        Sampler *uniform_sampler = new UniformSampler;
        Model *homography_model = new Model (3, 4, 0.99, 0, "homography");

        uniform_sampler->setSampleSize(homography_model->sample_number);
        uniform_sampler->setPointsSize(points1.rows);

        Ransac ransac (*homography_model, *uniform_sampler, *termination_criteria, *quality);
        ransac.run(points1, homograpy_estimator);

        cv::Mat H = ransac.best_model.returnDescriptor();

        std::ofstream save_model;
        std::string filename = "../results/homography/" + img_name.substr(0, img_name.find('_')) +"_Rmodel.txt";
        save_model.open(filename);

        save_model << H.at<float>(0,0) << " " << H.at<float>(0,1) << " " << H.at<float>(0,2) << '\n'
                   << H.at<float>(1,0) << " " << H.at<float>(1,1) << " " << H.at<float>(1,2) << '\n'
                   << H.at<float>(2,0) << " " << H.at<float>(2,1) << " " << H.at<float>(2,2) << '\n';

        save_model << ransac.most_inliers.size() << '\n';
        save_model << ransac.getQuality()->getIterations() << '\n';
        save_model << ransac.getQuality()->getComputationTime() << '\n';
        save_model.close();
    }
}