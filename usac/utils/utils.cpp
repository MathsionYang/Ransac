#include "utils.hpp"
#include "nearest_neighbors.hpp"

void densitySort (const cv::Mat &points, unsigned int knn, cv::Mat &sorted_points) {
    // get neighbors
    cv::Mat neighbors, neighbors_dists;
    NearestNeighbors::getNearestNeighbors_nanoflann(points, knn, neighbors, true, neighbors_dists);
    //

    std::vector<int> sorted_idx(points.rows);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    float sum1, sum2;
    int idxa, idxb;
    float *neighbors_dists_ptr = (float *) neighbors_dists.data;
    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) {
        sum1 = 0, sum2 = 0;
        idxa = knn * a, idxb = knn * b;
        for (int i = 0; i < knn; i++) {
            sum1 += neighbors_dists_ptr[idxa + i];
            sum2 += neighbors_dists_ptr[idxb + i];
        }
        return sum1 < sum2;
    });

    for (int i = 0; i < points.rows; i++) {
        sorted_points.push_back(points.row(sorted_idx[i]));
    }
}

void densitySort (const cv::Mat &points, unsigned int knn, cv::Mat &sorted_points, 
                 const std::vector<int> &inliers, std::vector<int> &sorted_inliers) {
    // get neighbors
    cv::Mat neighbors, neighbors_dists;
    NearestNeighbors::getNearestNeighbors_nanoflann(points, knn, neighbors, true, neighbors_dists);
    //

    std::vector<int> sorted_idx(points.rows);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    float sum1, sum2;
    int idxa, idxb;
    float *neighbors_dists_ptr = (float *) neighbors_dists.data;
    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) {
        sum1 = 0, sum2 = 0;
        idxa = knn * a, idxb = knn * b;
        for (int i = 0; i < knn; i++) {
            sum1 += neighbors_dists_ptr[idxa + i];
            sum2 += neighbors_dists_ptr[idxb + i];
        }
        return sum1 < sum2;
    });

    sorted_inliers.clear();
    sorted_inliers.reserve(inliers.size());
    
    for (int i = 0; i < points.rows; i++) {
        sorted_points.push_back(points.row(sorted_idx[i]));
    
        if (std::find(inliers.begin(), inliers.end(), sorted_idx[i]) != inliers.end()) {
            sorted_inliers.push_back(i);
        }
    }
}


/*
 * filename = path/name.ext
 * path
 * name
 * ext
 */
void splitFilename (const std::string &filename, std::string &path, std::string &name, std::string &ext) {
    const unsigned long dot = filename.find_last_of('.');
    const unsigned long slash = filename.find_last_of('/');
    // substr (pos, n) take substring of size n starting from position pos.
    path = filename.substr(0, slash+1);
    name = filename.substr(slash+1, dot-slash-1);
    ext = filename.substr(dot+1, filename.length()-1);
}

void solveLSQWithQR (cv::Mat& x, const cv::Mat& A, const cv::Mat& b) {
    // todo
    // solves Ax = b, A = QR
    // QRx = b |*Q' -> Rx = Q'b -> x = inv(R)*Q'b, R is upper triangular,
    // with back substitution will be fast. 
}



float quicksort_median (float * array, unsigned int k_minth, unsigned int left, unsigned int right) {
    unsigned int lenght = right - left;
    if (lenght == 0) {
        return array[left];
    }
    float pivot_val = array[right];
    int j;
    int right_ = right-1;
    unsigned int values_less_eq = 1;
    for (j = left; j <= right_;) {
        if (array[j] <= pivot_val) {
            j++;
            values_less_eq++;
        } else {
            float temp = array[j];
            array[j] = array[right_];
            array[right_] = temp;
            right_--;
        }
    }
    if (values_less_eq == k_minth) return pivot_val;
    if (k_minth > values_less_eq) {
        quicksort_median(array, k_minth - values_less_eq, j, right-1);
    } else {
        if (j == left) j++;
        quicksort_median(array, k_minth, left, j-1);
    }
}

// find median using quicksort with average O(n) complexity. Worst case is O(n^2).
float findMedian (float * array, unsigned int length) {
    if (length % 2 == 1) {
        // odd number of values
        return quicksort_median (array, length/2+1, 0, length-1);
    } else {
        // even: return average
        return (quicksort_median(array, length/2, 0, length-1) + quicksort_median(array, length/2+1, 0, length-1))/2;
    }
}
