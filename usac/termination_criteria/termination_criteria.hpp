#ifndef USAC_TERMINATIONCRITERIA_H
#define USAC_TERMINATIONCRITERIA_H

#include "../precomp.hpp"

class TerminationCriteria {
protected:
    bool isinit = false;
public:
    bool isInit () { return isinit; }
    virtual ~TerminationCriteria() = default;

    // use inline
    virtual unsigned int getUpBoundIterations (unsigned int inlier_size) = 0;
    virtual unsigned int getUpBoundIterations (unsigned int inlier_size, unsigned int points_size) = 0;
};
#endif //USAC_TERMINATIONCRITERIA_H
