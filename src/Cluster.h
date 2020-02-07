#ifndef CLUSTER_H
#define CLUSTER_H
#include <vector>

class Cluster{
 public: 
Cluster(std::vector<std::vector<float>*>* input) {};
  std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster();

};


#endif
