#ifndef ITERATIVECLUSTERING_H
#define ITERATIVECLUSTERING_H
#include "Cluster.h"
#include <vector>

class IterativeClustering{
 public:
 IterativeClustering(Cluster* cluster, std::vector<std::vector<float>*>* data) : c(cluster), data(data){};
  std::vector<std::pair<std::vector<std::vector<float>*>*,std::vector<bool>*>> findKClusters(int k) ;
  
  
 private:
  Cluster* c;
  std::vector<std::vector<float>*>* data;
  
};


#endif
