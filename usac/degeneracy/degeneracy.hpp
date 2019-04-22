#ifndef USAC_DEGENERACY_H
#define USAC_DEGENERACY_H

class Degeneracy {
public:
    virtual ~Degeneracy() = default;
    virtual void fix (const int * const sample, Model * best_model, Score * best_score) {}
};

#endif //USAC_DEGENERACY_H
