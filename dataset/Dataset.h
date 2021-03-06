#ifndef USAC_DATASET_H
#define USAC_DATASET_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

enum DATASET {Homogr, Homogr_SIFT, Adelaidermf, Adelaidermf_SIFT, Kusvod2, Kusvod2_SIFT, Syntectic, Strecha, EVD};
class Dataset {
public:
    static void saveDatasetToFiles ();
    static std::vector<std::string> getDataset (DATASET dataset) {
        if (dataset == DATASET::Homogr || dataset == DATASET::Homogr_SIFT) {
            return getHomographyDatasetPoints();
        } else if (dataset == DATASET::Adelaidermf || dataset == DATASET ::Adelaidermf_SIFT) {
            return getAdelaidermfDataset();
        } else if (dataset == DATASET::Kusvod2 || dataset == DATASET::Kusvod2_SIFT) {
            return getKusvod2Dataset();
        } else if (dataset == DATASET::Syntectic) {
            return getSyntecticLine2dDataset();
        } else if (dataset == DATASET::Strecha) {
            return getStrechaDataset();

        } else if (dataset == EVD) {
            return getEVDDataset();
        } else {
            std::cout << "Unknown dataset in class Dataset\n";
            exit(111);
        }
    }

    static std::vector<std::string> getEVDDataset ();
    static std::vector<std::string> getHomographyDatasetPoints ();
    static std::vector<std::string> getProblemHomographyDatasetPoints ();
    static std::vector<std::string> getStrechaCastleDenseImages (int option);
    static std::vector<std::string> getKusvod2Dataset ();
    static std::vector<std::string> getAdelaidermfDataset ();
    static std::vector<std::string> getStrechaDataset ();
    static std::vector<std::string> getSyntecticLine2dDataset ();
};

#endif //USAC_DATASET_H
