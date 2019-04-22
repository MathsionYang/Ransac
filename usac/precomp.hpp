#ifndef PRECOMP_H
#define PRECOMP_H

// C++
#include <iostream>
#include <cstdio>
#include <string>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <thread>
#include <vector>
#include <cassert>
#include <memory>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <iomanip>

// Nanoflann (only for nearest neighbors searching, maybe we should try to move nanoflann 
// source code to include folder and don't inlcude whole library)
#include "../include/nanoflann/nanoflann.hpp"
#include "../include/nanoflann/KDTreeVectorOfVectorsAdaptor.h"

// OpenCV
#include <opencv2/flann/flann.hpp>
#include <opencv2/flann/miniflann.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>

#endif // PRECOMP_H
