#ifndef READPOINTS_READPOINTS_H
#define READPOINTS_READPOINTS_H

#include <opencv2/core/mat.hpp>

void read_points (cv::Mat &pts1, cv::Mat &pts2, const std::string &filename);
void getInliers (const std::string &filename, std::vector<int> &inliers);
void getH (const std::string &filename, cv::OutputArray H);

#endif //READPOINTS_READPOINTS_H