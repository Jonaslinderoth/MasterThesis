#ifndef MINECLUS_H
#define MINECLUS_H

#include "../Clustering.h"
#include "../dataReader/DataReader.h"
#include <boost/dynamic_bitset.hpp>
#include "MuApriori/MuApriori.h"

class MineClus : public Clustering{
 public:
 MineClus() :MineClus(new std::vector<std::vector<float>*>){};
	MineClus(std::vector<std::vector<float>*>* input, float alpha = 0.1, float beta = 0.25, float width = 15, unsigned int d0 = 0);

	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster() override;
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> findKClusters(int k) override;

	
	float mu(int a, int b){
		return log(a)*log((float) 1/this->beta)*b;
	};
	void setAlpha(float value){this->alpha = value;};
	void setBeta(float value){this->beta = value;};
	void setWidth(float value){this->width = value;};
	void setd0(unsigned int value){this->d0 = value;};
	void setConcurentVersion(bool value){this->concurent = value;};
	bool isConcurentVersion(){return this->concurent;};
	unsigned int getd0(){return this->d0;};
	float getAlpha(){return this->alpha;};
	float getBeta(){return this->beta;};
	float getWidth(){return this->width;};
	std::vector<OutputCandidate*>* findClusterCandidates();

	
	
	unsigned int size(){return this->data->size();};
	std::vector<boost::dynamic_bitset<>>* findDimensions(std::vector<float>* centroid,
									  std::vector<std::vector<float>* >* points, float width);
	std::vector<std::vector<float>*>* pickRandom(int n);
	virtual ~MineClus(){
		delete this->medoids;
	};
	bool isDisjoint(OutputCandidate* lhs, OutputCandidate* rhs);
 	float alpha;
	float beta;
	float width;
	unsigned int d0;
	bool concurent=true;
	std::vector<std::vector<float>*>* data;
	std::vector<std::vector<float>*>* medoids;
	
};

#endif
