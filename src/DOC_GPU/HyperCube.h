#ifndef HYPERCUBE_GPU_H
#define HYPERCUBE_GPU_H
#include <vector>


std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> findDimmensions(std::vector<std::vector<float>*>* ps,
																					   std::vector<std::vector<std::vector<float>*>*> Xs,int m, float width = 10.0);


std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> pointsContained(std::vector<std::vector<bool>*>* dims,
												 std::vector<std::vector<float>*>* data,
																					   std::vector<std::vector<float>*>* centroids, int m,  float width = 10.0);


int argMax(std::vector<float>* scores);


std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster(std::vector<std::vector<float>*>* data, float alpha = 0.1, float beta = 0.25, float width = 15);


float* scoreHost(unsigned int* Cluster_size, unsigned int* Dim_count, float* score_output, int len, float beta);



#endif
