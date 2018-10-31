#include "Dataset.h"

void saveDatasetToFiles () {
    FILE *fp1, *fp2;
    std::string evd_path1 = "../dataset/EVD/1/";
    std::string evd_path2 = "../dataset/EVD/2/";

    std::string command1 = "ls "+evd_path1+"*.jpg "+evd_path1+"*.png "+evd_path1+"*.JPEG 2> /dev/null";
    std::string command2 = "ls "+evd_path2+"*.jpg "+evd_path2+"*.png "+evd_path2+"*.JPEG 2> /dev/null";

    fp1 = popen((command1).c_str(), "r");
    fp2 = popen((command2).c_str(), "r");

    if (fp1 == nullptr || fp2 == nullptr) {
        printf("EVD dataset not found\n" );
        exit(1);
    }

    FILE *evd1 = fopen("../dataset/evd1.txt", "w");
    FILE *evd2 = fopen("../dataset/evd2.txt", "w");

    char filename1[500], filename2[500];

    while (fgets(filename1, sizeof(filename1)-1, fp1) != nullptr && fgets(filename2, sizeof(filename2)-1, fp1) != nullptr) {
        fprintf(evd1, "%s", filename1);
        fprintf(evd2, "%s", filename2);
    }

    fclose(evd1); fclose(evd2);
    pclose(fp1); pclose(fp2);
}
/*
 * Returns image filenames
 * Vector n x 2
 * first column is first correspondence
 * second column is second correspondence
 */
std::vector<std::vector<std::string>> getEVDfilenames (bool reset_files) {
    if (reset_files) saveDatasetToFiles();

    std::vector<std::vector<std::string>> filenames (2);
    std::ifstream evd1 ("../dataset/evd1.txt");
    std::ifstream evd2 ("../dataset/evd2.txt");
    std::string fn1, fn2;
    while (std::getline(evd1, fn1) && std::getline(evd2, fn2)) {
        filenames[0].push_back(fn1);
        filenames[1].push_back(fn2);
    }
    return filenames;
};

std::vector<std::string> getHomographyDatasetPoints (bool reset_files) {
    std::vector<std::string> fnames = {"adam_pts.txt", "Brussels_pts.txt", "LePoint1_pts.txt",
                                       "boat_pts.txt",          "CapitalRegion_pts.txt",  "LePoint2_pts.txt",
                                       "BostonLib_pts.txt",     "city_pts.txt",           "LePoint3_pts.txt",
                                       "Boston_pts.txt",        "Eiffel_pts.txt",         "WhiteBoard_pts.txt",
                                       "BruggeSquare_pts.txt",  "ExtremeZoom_pts.txt",
                                       "BruggeTower_pts.txt",   "graf_pts.txt"};

    return fnames;
};


std::vector<std::string> getFundamentalDatasetPoints (bool reset_files) {
    std::vector<std::string> fnames = {"barrsmith_annot.txt", "barrsmith_pts.txt", "bonhall_pts.txt",
                                       "bonython_pts.txt", "elderhalla_pts.txt", "elderhallb_pts.txt",
                                       "hartley_pts.txt", "johnssona_pts.txt", "johnssonb_pts.txt",
                                       "ladysymon_pts.txt", "library_pts.txt", "napiera_pts.txt",
                                       "napierb_pts.txt", "neem_pts.txt", "unihouse_pts.txt",
                                       "oldclassicswing_pts.txt", "physics_pts.txt", "sene_pts.txt",
                                       "unionhouse_pts.txt"};
    return fnames;
};