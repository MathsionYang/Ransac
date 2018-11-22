#ifndef USAC_STATISTICALRESULTS_H
#define USAC_STATISTICALRESULTS_H

class StatisticalResults {
public:
    // -1 is not estimated yet
    // std_dev is standart deviation
    // avg is average

    long avg_time_mcs = 0;
    long median_time_mcs = 0;
    long std_dev_time_mcs = 0;

    float avg_num_inliers = 0;
    float median_num_inliers = 0;
    float std_dev_num_inliers = 0;

    float avg_avg_error = -1;
    float median_avg_error = -1;
    float std_dev_avg_error = -1;

    float avg_num_iters = 0;
    float median_num_iters = 0;
    float std_dev_num_iters = 0;

    int num_fails = -1;

    friend std::ostream& operator<< (std::ostream& stream, const StatisticalResults * res) {
        return stream
                << "Average time (mcs) " << res->avg_time_mcs << "\n"
                << "Standard deviation of time " << res->std_dev_time_mcs << "\n"
                << "Median of time " << res->median_time_mcs << "\n"
                << "-----------------\n"
                << "Average average error " << res->avg_avg_error << "\n"
                << "Standard deviation of average error " << res->std_dev_avg_error << "\n"
                << "Median of average error " << res->median_avg_error << "\n"
                << "-----------------\n"
                << "Average number of inliers " << res->avg_num_inliers << "\n"
                << "Standard deviation of number of inliers " << res->std_dev_num_inliers << "\n"
                << "Median of number of inliers " << res->median_num_inliers << "\n"
                << "-----------------\n"
                << "Average number of iterations " << res->avg_num_iters << "\n"
                << "Standard deviation of number of iterations " << res->std_dev_num_iters << "\n"
                << "Median of number of iterations " << res->median_num_iters << "\n"
                << "-----------------\n"
                << res->num_fails << " failed models\n";
    }

    void copyFrom (StatisticalResults * statisticalResults) {
        avg_time_mcs = statisticalResults->avg_time_mcs;
        median_time_mcs = statisticalResults->median_time_mcs;
        std_dev_time_mcs = statisticalResults->std_dev_time_mcs;

        avg_num_inliers = statisticalResults->avg_num_inliers;
        median_num_inliers = statisticalResults->median_num_inliers;
        std_dev_num_inliers = statisticalResults->std_dev_num_inliers;

        avg_avg_error = statisticalResults->avg_avg_error;
        median_avg_error = statisticalResults->median_avg_error;
        std_dev_avg_error = statisticalResults->std_dev_avg_error;

        avg_num_iters = statisticalResults->avg_num_iters;
        median_num_iters = statisticalResults->median_num_iters;
        std_dev_num_iters = statisticalResults->std_dev_num_iters;

        num_fails = statisticalResults->num_fails;
    }
};


#endif //USAC_STATISTICALRESULTS_H
