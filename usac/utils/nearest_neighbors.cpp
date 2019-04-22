#include "../precomp.hpp"

#include "nearest_neighbors.hpp"
#include "../../detector/Reader.h"
#include "math.hpp"

/*
 * Problem with repeated points in flann and nanoflann:
 *

0 38 37 39 42 20 21
1 32 2 30 34 33 47
2 30 32 1 34 33 31
3 9 43 11 15 14 29
4 20 21 38 37 39 16
5 6 51 29 18 19 9
6 5 51 29 18 19 9
7 29 47 5 6 3 9
8 23 49 48 35 52 22
9 43 3 11 15 14 29
10 13 15 14 50 11 9
11 15 14 9 43 3 13
12 44 45 25 24 27 0
13 10 15 14 50 11 9
15 14 11 13 9 43 3 // here must be 14 15
15 14 11 13 9 43 3
 */

/*
[0, 37, 38, 39, 42, 20, 21, 4;
 1, 32, 2, 30, 33, 34, 47, 31;
 2, 30, 32, 1, 33, 34, 31, 46;
 3, 9, 43, 11, 14, 15, 29, 13;
 4, 20, 21, 37, 38, 39, 16, 17;
 5, 6, 51, 29, 18, 19, 9, 43;
 6, 5, 51, 29, 18, 19, 9, 43;
 7, 29, 47, 5, 6, 3, 9, 43;
 8, 23, 49, 48, 35, 52, 22, 28;
 9, 43, 3, 11, 14, 15, 29, 13;
 10, 13, 14, 15, 50, 11, 9, 43;
 11, 14, 15, 9, 43, 3, 13, 29;
 12, 44, 45, 24, 25, 27, 0, 20;
 13, 10, 14, 15, 50, 11, 9, 43;
 14, 15, 11, 13, 9, 43, 3, 10;
 14, 15, 11, 13, 9, 43, 3, 10; // must be 15 14
 16, 17, 4, 20, 21, 18, 19, 37;
 */




/*
 * @points N x 2
 * x1 y1
 * ...
 * xN yN
 *
 * @k_nearest_neighbors is number of nearest neighbors for each point.
 *
 * @nearest_neighbors is matrix N x k of indexes of nearest points
 * x1_nn1 x1_nn2 ... x1_nnk
 * ...
 * xN_nn1 xN_nn2 ... xN_nnk
 */
void NearestNeighbors::getNearestNeighbors_nanoflann (const cv::Mat& points, int k_nearest_neighbors,
                                                      cv::Mat &nearest_neighbors, bool get_distances,
                                                      cv::Mat &nearest_neighbors_distances) {
    unsigned int points_size = points.rows;
    unsigned int dim = points.cols;
    
    std::vector<std::vector<float>> samples (points_size, std::vector<float>(dim));
    
    for (unsigned int p = 0; p < points_size; p++) {
        points.row(p).copyTo(samples[p]);
    }

    // construct a kd-tree index:
    // Dimensionality set at run-time (default: L2)
    // ------------------------------------------------------------
    typedef KDTreeVectorOfVectorsAdaptor< std::vector<std::vector<float> >, float >  my_kd_tree_t;

    my_kd_tree_t   mat_index(dim /*dim*/, samples, 10 /* max leaf */ );
    mat_index.index->buildIndex();


    // do a knn search
    unsigned long ret_indexes[k_nearest_neighbors + 1];
    float out_dists_sqr[k_nearest_neighbors + 1];
    nanoflann::KNNResultSet<float> resultSet(k_nearest_neighbors + 1);

    nearest_neighbors = cv::Mat_<int>(points_size, k_nearest_neighbors);
    int *nearest_neighbors_ptr = (int *) nearest_neighbors.data;

    int p_idx;
    if (get_distances) {
        nearest_neighbors_distances = cv::Mat_<float>(points_size, k_nearest_neighbors);
        float *nearest_neighbors_distances_ptr = (float *) nearest_neighbors_distances.data;

        for (unsigned int p = 0; p < points_size; p++) {
            resultSet.init(ret_indexes, out_dists_sqr);
            mat_index.index->findNeighbors(resultSet, &samples[p][0], nanoflann::SearchParams(10));
            p_idx = k_nearest_neighbors * p;

            for (int nn = 0; nn < k_nearest_neighbors; nn++) {
                nearest_neighbors_ptr[p_idx + nn] = (int) ret_indexes[nn + 1];
                nearest_neighbors_distances_ptr[p_idx + nn] = out_dists_sqr[nn + 1];
            }
        }
    } else {
        for (unsigned int p = 0; p < points_size; p++) {
            resultSet.init(ret_indexes, out_dists_sqr);
            mat_index.index->findNeighbors(resultSet, &samples[p][0], nanoflann::SearchParams(10));
            p_idx = k_nearest_neighbors * p;
            for (int nn = 0; nn < k_nearest_neighbors; nn++) {
                nearest_neighbors_ptr[p_idx + nn] = ret_indexes[nn + 1];
            }
        }
    }
}

/*
 * @points N x 2
 * x1 y1
 * ...
 * xN yN
 *
 * @k_nearest_neighbors is number of nearest neighbors for each point.
 *
 * @nearest_neighbors is matrix N x k of indexes of nearest points
 * x1_nn1 x1_nn2 ... x1_nnk
 * ...
 * xN_nn1 xN_nn2 ... xN_nnk
 *
 */
void NearestNeighbors::getNearestNeighbors_flann (const cv::Mat& points, int k_nearest_neighbors, cv::Mat &nearest_neighbors) {
    unsigned int points_size = points.rows;
    cv::flann::LinearIndexParams flannIndexParams;
    cv::flann::Index flannIndex (points.reshape(1), flannIndexParams);
    cv::Mat dists;

    flannIndex.knnSearch(points, nearest_neighbors, dists, k_nearest_neighbors+1);

    // first nearest neighbor of point is this point itself.
    // remove this first column
    nearest_neighbors.colRange(1, k_nearest_neighbors+1).copyTo (nearest_neighbors);

//    std::cout << nearest_neighbors << "\n\n";
//    std::cout << dists << "\n\n";
}

void NearestNeighbors::getGridNearestNeighbors (const cv::Mat& points, int cell_sz, std::vector<std::vector<int>> &neighbors) {
    // cell size 25, 50, 100
    std::map<CellCoord, std::vector<int>> neighbors_map;
//    std::unordered_map<CellCoord, std::vector<int>, hash_fn> neighbors_map;

    float *points_p = (float *) points.data;
    unsigned int idx, points_size = points.rows;
    CellCoord c;
    neighbors = std::vector<std::vector<int>>(points_size);
    for (unsigned int i = 0; i < points_size; i++) {
        neighbors[i].reserve(10); // reserve predicted neighbors size

        idx = 4 * i;
        c.init(points_p[idx] / cell_sz, points_p[idx + 1] / cell_sz, points_p[idx + 2] / cell_sz,
               points_p[idx + 3] / cell_sz);
        neighbors_map[c].push_back(i);

    }

    // debug
//    for (auto cells : neighbors_map) {
//        std::cout << "key = (" << cells.first.c1x << " " << cells.first.c1y << " " << cells.first.c2x << " "
//                  << cells.first.c2y <<
//                  ") values = ";
//        for (auto v : cells.second) {
//            std::cout << v << " ";
//        }
//        std::cout << "\n";
//    }

    unsigned long neighbors_in_cell;
    for (auto cells : neighbors_map) {
        neighbors_in_cell = cells.second.size();
        if (neighbors_in_cell < 2) continue;

        for (unsigned int n1 = 0; n1 < neighbors_in_cell; n1++) {
            for (unsigned int n2 = n1+1; n2 < neighbors_in_cell; n2++) {
                neighbors[cells.second[n1]].push_back(cells.second[n2]);
                neighbors[cells.second[n2]].push_back(cells.second[n1]);
            }
        }
    }
}