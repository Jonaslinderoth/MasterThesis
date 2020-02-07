#ifndef CLUSTER_H
#define CLUSTER_H
#include <vector>
#include <iostream>

class Cluster{
 public: 
Cluster(std::vector<std::vector<float>*>* input) {};
 virtual std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster() {};
 virtual void setData(std::vector<std::vector<float>*>* input) {
 };
};

#endif
