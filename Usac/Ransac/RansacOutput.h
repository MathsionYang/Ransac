#ifndef USAC_RANSACOUTPUT_H
#define USAC_RANSACOUTPUT_H

#include "../Model.h"

struct Time {
    long minutes;
    long seconds;
    long milliseconds;
    long microseconds;
};

class RansacOutput {
private:
    Model * model;
    Time * time;
    std::vector<int> inliers;
    long time_mcs;
    unsigned int number_inliers;
    unsigned int number_iterations;
    unsigned int number_lo_iterations;
    unsigned int lo_runs;

public:

   ~RansacOutput() {
       delete model, time;
   }

    RansacOutput (const Model * const model_,
                  const int * const inliers_,
                  long time_mcs_,
                  unsigned int number_inliers_,
                  unsigned int number_iterations_,
                  unsigned int number_lo_iterations_,
                  unsigned int lo_runs_) {

        /*
         * Let's make a deep copy to avoid changing variables from origin input.
         * And make them changeable for further using.
         */
    
        model = new Model (*model_);        
        inliers.assign (inliers_, inliers_ + number_inliers_);
        time_mcs = time_mcs_;
        number_inliers = number_inliers_;
        number_iterations = number_iterations_;
        number_lo_iterations = number_lo_iterations_;
        lo_runs = lo_runs_;

        time = new Time;
        time->microseconds = time_mcs % 1000;
        time->milliseconds = ((time_mcs - time->microseconds)/1000) % 1000;
        time->seconds = ((time_mcs - 1000*time->milliseconds - time->microseconds)/(1000*1000)) % 60;
        time->minutes = ((time_mcs - 60*1000*time->seconds - 1000*time->milliseconds - time->microseconds)/(60*1000*1000)) % 60;

    }

    void printTime () {
        std::cout << time->seconds << " secs, " << time->milliseconds << " ms, " << time->microseconds << " mcs\n";
    }

    std::vector<int> getInliers () {
        return inliers;
    }

    long getTimeMicroSeconds () {
        return time_mcs;
    }

    unsigned int getNumberOfInliers () {
        return number_inliers;
    }

    unsigned int getNumberOfIterations () {
        // number_iterations > number_lo_iterations
        return number_iterations;
    }

    unsigned int getNumberOfLOIterations () {
        // number_iterations > number_lo_iterations
        return number_lo_iterations;
    }
    
    unsigned int getLORuns () {
        return lo_runs;
    }
    Time* getTime () {
        return time;
    }

    Model* getModel () {
        return model;
    }

};

#endif //USAC_RANSACOUTPUT_H