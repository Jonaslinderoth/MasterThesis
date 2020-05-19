#ifndef FAST_DOC_H
#define FAST_DOC_H

#include <vector>
#include <math.h>
#include "../Clustering.h"
#include "../dataReader/DataReader.h"

class Fast_DOC : public Clustering{
 public:
 Fast_DOC() :Fast_DOC(new std::vector<std::vector<float>*>){};
	//Fast_DOC(DataReader* dr);
	Fast_DOC(std::vector<std::vector<float>*>* input, float alpha = 0.1, float beta = 0.25, float width = 15, unsigned int d0 = 0);
 Fast_DOC(DataReader* dr) : Fast_DOC(initDataReader(dr)){};
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster() override;
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> findKClusters(int k) override;

	
	
	float mu(int a, int b){
		return log(a)+log((float) 1/this->beta)*b;
	};
	void setAlpha(float value){this->alpha = value;};
	void setBeta(float value){this->beta = value;};
	void setWidth(float value){this->width = value;};
	void setd0(unsigned int value){this->d0 = value;};
	unsigned int getd0(){return this->d0;};
	float getAlpha(){return this->alpha;};
	float getBeta(){return this->beta;};
	float getWidth(){return this->width;};
	
	unsigned int size(){return this->data->size();};
	std::vector<bool>* findDimensions(std::vector<float>* centroid,
									  std::vector<std::vector<float>* >* points, float width);
	std::vector<std::vector<float>*>* pickRandom(int n);
	virtual ~Fast_DOC(){};
 private:
	
 	float alpha;
	float beta;
	float width;
	unsigned int d0;
	std::vector<std::vector<float>*>* initDataReader(DataReader* dr);
	std::vector<std::vector<float>*>* data;
	
};

#endif
