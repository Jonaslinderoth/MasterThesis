#ifndef MINECLUSKERNELS_H
#define MINECLUSKERNELS_H
#include <vector>

std::vector<unsigned int> createItemSetTester(std::vector<std::vector<float>*>* data, unsigned int centroid, float width);
std::vector<unsigned int> createInitialCandidatesTester(unsigned int dim);
#endif
