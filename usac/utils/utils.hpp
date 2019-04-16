// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef USAC_UTILS_H
#define USAC_UTILS_H

#include "../precomp.hpp"

void densitySort (const cv::Mat &points, unsigned int max_neighbor, cv::Mat &sorted_points);
void densitySort (const cv::Mat &points, unsigned int knn, cv::Mat &sorted_points, 
                 const std::vector<int> &inliers, std::vector<int> &sorted_inliers);

void splitFilename (const std::string &filename, std::string &path, std::string &name, std::string &ext);

float quicksort_median (float * array, unsigned int k_minth, unsigned int left, unsigned int right);
float findMedian (float * array, unsigned int length);

void solveLSQWithQR (cv::Mat& x, const cv::Mat& A, const cv::Mat& b);

#endif //USAC_UTILS_H
