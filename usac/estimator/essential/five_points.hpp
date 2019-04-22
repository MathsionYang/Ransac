#ifndef USAC_FIVEPOINTSALGORITHM_H
#define USAC_FIVEPOINTSALGORITHM_H

#include "../../precomp.hpp"

class EssentialSolver {
private:
	const float * const points;
public:
	EssentialSolver (const float * const points_) : points (points_) {}
	
	unsigned int FivePoints (const int * const sample, cv::Mat &E);
};

static void ProjectionsFromEssential(const cv::Mat &E, cv::Mat &P1, cv::Mat &P2, cv::Mat &P3, cv::Mat &P4);
static cv::Mat TriangulatePoint(const cv::Point2d &pt1, const cv::Point2d &pt2, const cv::Mat &P1, const cv::Mat &P2);
static double CalcDepth(const cv::Mat &X, const cv::Mat &P);
static bool Solve5PointEssential(std::vector<cv::Point2d> &pts1, std::vector<cv::Point2d> &pts2, cv::Mat &ret_E, cv::Mat &ret_P);


#endif //USAC_FIVEPOINTSALGORITHM_H