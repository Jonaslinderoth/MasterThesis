#ifndef MINECLUSKERNELS_H
#define MINECLUSKERNELS_H
#include <vector>
#include <tuple>

std::vector<unsigned int> createItemSetTester(std::vector<std::vector<float>*>* data, unsigned int centroid, float width);
std::vector<unsigned int> createInitialCandidatesTester(unsigned int dim);
std::tuple<std::vector<unsigned int>,std::vector<float>, std::vector<bool>> countSupportTester(std::vector<std::vector<bool>> candidates, std::vector<std::vector<bool>> itemSet, unsigned int minSupp, float beta);
std::vector<unsigned int> mergeCandidatesTester(std::vector<std::vector<bool>> candidates);
#endif
