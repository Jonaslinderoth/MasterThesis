#include "IterativeClustering.h"
#include "Cluster.h"


std::vector<std::pair<std::vector<std::vector<float>*>*,std::vector<bool>*>> IterativeClustering::findKClusters(int k){
  // TODO: sub optimal
  
  this->c->setData(this->data);
    
    std::vector<std::pair<std::vector<std::vector<float>*>*,std::vector<bool>*>> result = std::vector<std::pair<std::vector<std::vector<float>*>*,std::vector<bool>*>>{};

  for(int i = 0; i<k; i++){

    auto currentCluster = this->c->findCluster();
    result.push_back(currentCluster);

    int frontOfCluster = 0;
    for(int j = 0; j < this->data->size(); j++){
      std::vector<float>* a = this->data->at(j);
      std::vector<float>* b = currentCluster.first->at(frontOfCluster);
      if(a == currentCluster.first->at(frontOfCluster)){
	frontOfCluster++;
	this->data->erase(this->data->begin()+j);
	j--;
      }
      

    }
  }
  return result;
  
}
