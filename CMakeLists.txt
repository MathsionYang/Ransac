cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
# comment or change
SET(OpenCV_DIR /home/miv1774/opencv-3.4.0/build/lib)

project (usac)
find_package (OpenCV REQUIRED)

add_executable(
        ransac main.cpp

        usac/ransac/ransac.cpp
        usac/ransac/init.cpp
                
        usac/estimator/dlt/dlt.cpp
        usac/estimator/dlt/normalized_dlt.cpp
        usac/estimator/dlt/normalizing_transformation.cpp
        usac/estimator/essential/five_points.cpp
        usac/estimator/essential/rpoly.cpp 
        usac/estimator/fundamental/seven_points.cpp
        usac/estimator/fundamental/eight_points.cpp
        
        usac/local_optimization/graphcut.cpp 

        usac/utils/math.cpp 
        usac/utils/nearest_neighbors.cpp 
        usac/utils/utils.cpp

        generator/generator.cpp

        detector/Reader.cpp
        detector/detector.cpp
        
        helper/drawing/DrawPanorama.cpp
        helper/drawing/DrawEpipolarLines.cpp

        dataset/Dataset.cpp
        dataset/SaveGTModel.cpp

        test/test_homography_fitting.cpp
        test/test_line2d_fitting.cpp
        test/test_essential_fitting.cpp
        test/test_fundamental_fitting.cpp
        test/test_affine_fitting.cpp
        test/test.cpp
        test/test_neighbors_search.cpp
        test/test_utils.cpp
        test/test_density_prosac.cpp
)

# parallel
# set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -pthread -ggdb" )
# warnings
# set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -pthread -Wextra" )
# optimalization
set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3" )

target_link_libraries (ransac ${OpenCV_LIBS} )

# theia
#find_package(Theia REQUIRED)
#include_directories(${THEIA_INCLUDE_DIRS})
#target_link_libraries (ransac ${THEIA_LIBRARIES})

# Nanoflann
# find_package(nanoflann)
# target_link_libraries(ransac nanoflann::nanoflann)

# Eigen
# find_package (Eigen3 3.3 REQUIRED NO_MODULE)
# target_link_libraries (ransac Eigen3::Eigen)