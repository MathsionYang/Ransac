#ifndef USAC_FUNDAMENTAL_DEGENERACY_H
#define USAC_FUNDAMENTAL_DEGENERACY_H

#include "../../dataset/GetImage.h"
#include "degeneracy.hpp"
#include "../utils/math.hpp"
#include "../estimator/homography_estimator.hpp"
#include "../ransac/init.hpp"

class FundamentalDegeneracy : public Degeneracy {
    const float * const points;
    unsigned int h_sample[5][3] = {{1,2,3},{4,5,6},{1,2,7},{4,5,7},{3,6,7}};
    HomographyEstimator homogr_est;
    float threshold;
    unsigned int points_size;
    unsigned int * outliers;
    Quality * quality;
    Score * score;
public:
    ~FundamentalDegeneracy() override {
        delete[] outliers;
    }

    FundamentalDegeneracy (cv::InputArray points_, Quality * quality_, float threshold_, SCORE score_) : points((float *)points_.getMat().data), homogr_est(points_) {
        points_size = points_.getMat().rows;
        threshold = threshold_;
        outliers = new unsigned int[points_size];
        quality = quality_;
//        Init::initScore(score, score_);
    }

    void fix (const int * const sample, Model * best_model, Score * best_score) override {

//        // debug show sample
//        ImageData img_data (DATASET::Adelaidermf, "barrsmith");
//        cv::Mat img1 = img_data.getImage1();
//        for (int i = 0; i < 7; i++) {
//            cv::circle(img1, cv::Point(points[4*sample[i]], points[4*sample[i]+1]),  5, cv::Scalar(255, 0, 0), -1);
//        }
//        //
//        cv::imshow("sample", img1);
//        cv::waitKey(0);
//        cv::destroyAllWindows();

        // According to Two-view Geometry Estimation Unaffected by a Dominant Plane
        // (http://cmp.felk.cvut.cz/~matas/papers/chum-degen-cvpr05.pdf)
        // only 5 homographies enough to test
        // triplets{1,2,3},{4,5,6},{1,2,7},{4,5,7}and{3,6,7}

        // H = A - e' (M^-1 b)^T
        // A = [e']_x F
        // b_i = (x′i × (A xi))^T (x′i × e′)‖x′i×e′‖^−2,
        // M is a 3×3 matrix with rows x^T_i
        // epipole e' is left nullspace of F s.t. e′^T F=0,
        // or F^T e' = 0 -> SVD(F^T): last column of V is null space e'.

        cv::Mat F = best_model->returnDescriptor();

        // find e'
        cv::Mat w, u, vt;
        cv::SVD::compute(F.t(), w, u, vt);
        cv::Mat e_prime = vt.row(vt.rows-1).t();
        //

        cv::Mat A = Math::getSkewSymmetric(e_prime) * F;

        cv::Mat H;
        cv::Mat_<float> xi_prime(3,1), xi(3,1), b(3,1), M(3,3);
        M.at<float>(0, 2) = 1;
        M.at<float>(1, 2) = 1;
        M.at<float>(2, 2) = 1;

        xi.at<float>(2) = 1;
        xi_prime.at<float>(2) = 1;

        bool model_is_degenerate = false;
        int smpl, inliers_on_plane;
        for (unsigned int h_i = 0; h_i < 5; h_i++) {
            for (unsigned int pt_i = 0; pt_i < 3; pt_i++) {
                // find b and M
                smpl = 4*sample[h_sample[h_i][pt_i]];
                xi.at<float>(0) = points[smpl];
                xi.at<float>(1) = points[smpl+1];
                xi_prime.at<float>(0) = points[smpl+2];
                xi_prime.at<float>(1) = points[smpl+3];

                // (x′i × e')
                cv::Mat xprime_X_eprime = Math::getCrossProductDim3(xi_prime, e_prime);
                // ||x′i × e'||
                float norm_xprime_X_eprime = cv::norm(xprime_X_eprime);

                // (x′i × (A xi))
                cv::Mat xprime_X_Ax = Math::getCrossProductDim3(xi_prime, A * xi);

                // x′i × (A xi))^T (x′i × e′) / ‖x′i×e′‖^2,
                b.at<float>(pt_i) = xprime_X_Ax.dot(xprime_X_eprime) / (norm_xprime_X_eprime * norm_xprime_X_eprime);

                // M from x^T
                M.at<float>(pt_i, 0) = xi.at<float>(0);
                M.at<float>(pt_i, 1) = xi.at<float>(1);
            }

            // compute H
            H = A - e_prime * (M.inv() * b).t();
            inliers_on_plane = 0;
            homogr_est.setModelParameters(H);
            // find inliers, points related to H, x' ~ Hx
            for (unsigned int s = 0; s < 7; s++) {
                float err = homogr_est.GetError(sample[s]);
//                std::cout << "err = " << err << "\n";
                if (err < threshold) {
                    inliers_on_plane++;
                }
            }
//            std::cout << "num inliers " << inliers_on_plane << "\n";
            // if there are inliers, so F is degenerate
            if (inliers_on_plane >= 3) { // 5 a lot?
                model_is_degenerate = true;
                break;
            }
        }

        std::cout << "model is degenerate " << model_is_degenerate << "\n";
        if (! model_is_degenerate) return;

        cv::Mat newF;
        planeAndParallax(H, newF);

        Score * temp;
//        initScore (temp, best_model->score);
        quality->getScore(temp, newF);
        if (temp->better(best_score)) {
            best_score->copyFrom(temp);
            best_model->setDescriptor(newF);
        }
    }

    void planeAndParallax (const cv::Mat &H, cv::Mat &F) {
        // find outliers of H
        unsigned int num_outliers = 0;
        for (unsigned int i = 0; i < points_size; i++) {
            if (homogr_est.GetError(i) >= threshold) {
                outliers[num_outliers++] = i;
            }
        }

        if (num_outliers < 2) return;

        // select 2 outliers at random
        unsigned int outlier1 = random() % num_outliers;
        unsigned int outlier2 = random() % num_outliers;
        while (outlier1 == outlier2) {
            outlier2 = random() % num_outliers;
        }

        cv::Mat pt1 = (cv::Mat_<float>(3,1) << points[4*outlier1], points[4*outlier1+1], 1);
        cv::Mat pt2 = (cv::Mat_<float>(3,1) << points[4*outlier2], points[4*outlier2+1], 1);
        cv::Mat pt1_prime = (cv::Mat_<float>(3,1) << points[4*outlier1+2], points[4*outlier1+3], 1);
        cv::Mat pt2_prime = (cv::Mat_<float>(3,1) << points[4*outlier2+2], points[4*outlier2+3], 1);

        cv::Mat l1 = Math::getCrossProductDim3(pt1_prime, H * pt1);
        cv::Mat l2 = Math::getCrossProductDim3(pt2_prime, H * pt2);
        cv::Mat ep = Math::getCrossProductDim3(l1, l2);
        cv::Mat skew_sym = Math::getSkewSymmetric(ep);
        F = skew_sym * H;
    }
};

#endif //USAC_FUNDAMENTAL_DEGENERACY_H
