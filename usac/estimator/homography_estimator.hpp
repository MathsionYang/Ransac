// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef RANSAC_HOMOGRAPHYESTIMATOR_H
#define RANSAC_HOMOGRAPHYESTIMATOR_H

#include "estimator.hpp"
#include "dlt/dlt.hpp"
#include "../utils/math.hpp"

class HomographyEstimator : public Estimator{
private:
    const float * const points;
    float h11, h12, h13, h21, h22, h23, h31, h32, h33;
    unsigned int points_size;
    DLt dlt;
public:
    ~HomographyEstimator () override = default;

    /*
     * @input_points: is matrix of size: number of points x 4
     * x1 y1 x'1 y'1
     * ...
     * xN yN x'N y'N
     */
    HomographyEstimator(cv::InputArray input_points) : points((float *)input_points.getMat().data), dlt(points) {
        assert(!input_points.empty());
        points_size = input_points.getMat().rows;
    }

    void setModelParameters (const cv::Mat& model) override {
        auto * H_ptr = (float *) model.data;

        h11 = H_ptr[0]; h12 = H_ptr[1]; h13 = H_ptr[2];
        h21 = H_ptr[3]; h22 = H_ptr[4]; h23 = H_ptr[5];
        h31 = H_ptr[6]; h32 = H_ptr[7]; h33 = H_ptr[8];
    }

    unsigned int EstimateModel(const int * const sample, std::vector<Model>& models) override {
//        if (! isSubsetGood(sample)){
//            return 0;
//        }

        cv::Mat H;
        if (! dlt.DLT4p (sample, H)) {
            return 0;
        }

        models[0].setDescriptor(H);

        return 1;
    }

    bool EstimateModelNonMinimalSample (const int * const sample, unsigned int sample_size, Model &model) override {
        cv::Mat H;
        if (! dlt.NormalizedDLT(sample, sample_size, H)) {
            return false;
        }

        model.setDescriptor(H);
        return true;
    }

    bool EstimateModelNonMinimalSample (const int * const sample, unsigned int sample_size, const float * const weights, Model &model) override {
        cv::Mat H;
        if (! dlt.NormalizedDLT(sample, sample_size, weights, H)) {
            return false;
        }

        model.setDescriptor(H);
        return true;
    }

    bool EstimateModelNonMinimalSample(const int * const sample, unsigned int sample_size, const float * const weightsx, const float * const weightsy, Model &model) override {
        cv::Mat H;
        if (! dlt.NormalizedDLT(sample, sample_size, weightsx, weightsy, H)) {
            return false;
        }

        model.setDescriptor(H);
        return true;
    }

    bool EstimateModelNonMinimalSample(const int * const sample, unsigned int sample_size,
                                               const float * const weightsx1, const float * const weightsy1,
                                               const float * const weightsx2, const float * const weightsy2, Model &model) override {
        cv::Mat H;
        if (! dlt.NormalizedDLT(sample, sample_size, weightsx1, weightsy1, weightsx2, weightsy2, H)) {
            return false;
        }

        model.setDescriptor(H);
        return true;
    }

    void getWeights (float * weights_euc1, float * weights_euc2, float * weights_manh1,
                     float * weights_manh2, float * weights_manh3, float * weights_manh4) override {
        cv::Mat H = (cv::Mat_<float>(3,3) << h11, h12, h13, h21, h22, h23, h31, h32, h33);
        H = H.inv();
        float * H_ptr = (float *) H.data;
        float h_inv11, h_inv12, h_inv13, h_inv21, h_inv22, h_inv23, h_inv31, h_inv32, h_inv33;
        h_inv11 = H_ptr[0]; h_inv12 = H_ptr[1]; h_inv13 = H_ptr[2];
        h_inv21 = H_ptr[3]; h_inv22 = H_ptr[4]; h_inv23 = H_ptr[5];
        h_inv31 = H_ptr[6]; h_inv32 = H_ptr[7]; h_inv33 = H_ptr[8];

        float x1, y1, x2, y2, est_x2, est_y2, est_z2, est_x1, est_y1, est_z1, error;
        unsigned int smpl;
        for (unsigned int pt = 0; pt < points_size; pt++) {
            smpl = 4*pt;
            x1 = points[smpl];
            y1 = points[smpl+1];
            x2 = points[smpl+2];
            y2 = points[smpl+3];

            est_x2 = h11 * x1 + h12 * y1 + h13;
            est_y2 = h21 * x1 + h22 * y1 + h23;
            est_z2 = h31 * x1 + h32 * y1 + h33; // h33 = 1

            est_x2 /= est_z2;
            est_y2 /= est_z2;

            est_x1 = h_inv11 * x2 + h_inv12 * y2 + h_inv13;
            est_y1 = h_inv21 * x2 + h_inv22 * y2 + h_inv23;
            est_z1 = h_inv31 * x2 + h_inv32 * y2 + h_inv33;

            est_x1 /= est_z1;
            est_y1 /= est_z1;

            float euc1 = sqrt ((x2 - est_x2) * (x2 - est_x2) + (y2 - est_y2) * (y2 - est_y2));
            float euc2 = sqrt ((x1 - est_x1) * (x1 - est_x1) + (y1 - est_y1) * (y1 - est_y1));

            float manh1 = fabsf(x1 - est_x1);
            float manh2 = fabsf(y1 - est_y1);
            float manh3 = fabsf(x2 - est_x2);
            float manh4 = fabsf(y2 - est_y2);

            weights_euc1[pt] = euc1 < 1 ? 1 : 1 / euc1;
            weights_euc2[pt] = euc2 < 1 ? 1 : 1 / euc2;

            weights_manh1[pt] = manh1 < 1 ? 1 : 1 / manh1;
            weights_manh2[pt] = manh2 < 1 ? 1 : 1 / manh2;
            weights_manh3[pt] = manh3 < 1 ? 1 : 1 / manh3;
            weights_manh4[pt] = manh4 < 1 ? 1 : 1 / manh4;
        }
    }

    bool LeastSquaresFitting (const int * const sample, unsigned int sample_size, Model &model) override {
        return EstimateModelNonMinimalSample(sample, sample_size, model);
    }
    /*
     * Error = distance (H pt(i), pt'(i))
     */
    float GetError(unsigned int pidx) override {
        unsigned int smpl = 4*pidx;
        float x1 = points[smpl];
        float y1 = points[smpl+1];
        float x2 = points[smpl+2];
        float y2 = points[smpl+3];

        float est_x2 = h11 * x1 + h12 * y1 + h13;
        float est_y2 = h21 * x1 + h22 * y1 + h23;
        float est_z2 = h31 * x1 + h32 * y1 + h33; // h33 = 1

        est_x2 /= est_z2;
        est_y2 /= est_z2;
        
        float error = sqrt ((x2 - est_x2) * (x2 - est_x2) + (y2 - est_y2) * (y2 - est_y2));
        // error >= 0
        return error;
    }

    unsigned int GetNumInliers (float threshold, bool get_inliers, int * inliers) override {
        float x1, y1, x2, y2, est_x2, est_y2, est_z2, error;
        unsigned int smpl;
        int num_inliers = 0;
        for (unsigned int pt = 0; pt < points_size; pt++) {
            smpl = 4*pt;
            x1 = points[smpl];
            y1 = points[smpl+1];
            x2 = points[smpl+2];
            y2 = points[smpl+3];

            est_x2 = h11 * x1 + h12 * y1 + h13;
            est_y2 = h21 * x1 + h22 * y1 + h23;
            est_z2 = h31 * x1 + h32 * y1 + h33; // h33 = 1

            est_x2 /= est_z2;
            est_y2 /= est_z2;
            
            error = sqrt ((x2 - est_x2) * (x2 - est_x2) + (y2 - est_y2) * (y2 - est_y2));
            if (error < threshold) {
                if (get_inliers) inliers[num_inliers] = pt;
                num_inliers++;
            }
        }

        return num_inliers;
    }

    bool isSubsetGood (const int * const sample) override {
        if (Math::haveCollinearPoints(points, sample, 4)) {
            std::cout << "points are collinear!\n";
            return false;
        }
        if (Math::isPointsClosed(points, sample, 4)) {
            std::cout << "points are closed!\n";
            return false;
        }

        return true;

//        // We check whether the minimal set of points for the homography estimation
//        // are geometrically consistent. We check if every 3 correspondences sets
//        // fulfills the constraint.
//        //
//        // The usefullness of this constraint is explained in the paper:
//        //
//        // "Speeding-up homography estimation in mobile devices"
//        // Journal of Real-Time Image Processing. 2013. DOI: 10.1007/s11554-012-0314-1
//        // Pablo Marquez-Neila, Javier Lopez-Alberca, Jose M. Buenaposada, Luis Baumela
//        if( count == 4 )
//        {
//            static const int tt[][3] = {{0, 1, 2}, {1, 2, 3}, {0, 2, 3}, {0, 1, 3}};
//            const Point2f* src = ms1.ptr<Point2f>();
//            const Point2f* dst = ms2.ptr<Point2f>();
//            int negative = 0;
//
//            for( int i = 0; i < 4; i++ )
//            {
//                const int* t = tt[i];
//                Matx33d A(src[t[0]].x, src[t[0]].y, 1., src[t[1]].x, src[t[1]].y, 1., src[t[2]].x, src[t[2]].y, 1.);
//                Matx33d B(dst[t[0]].x, dst[t[0]].y, 1., dst[t[1]].x, dst[t[1]].y, 1., dst[t[2]].x, dst[t[2]].y, 1.);
//
//                negative += determinant(A)*determinant(B) < 0;
//            }
//            if( negative != 0 && negative != 4 )
//                return false;
//        }

        return true;
    };
    int SampleNumber() override {
        return 4;
    }
};


#endif //RANSAC_HOMOGRAPHYESTIMATOR_H