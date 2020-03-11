#ifndef FAST_DOCGPU_H
#define FAST_DOCGPU_H
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include "../dataReader/DataReader.h"
#include "../Clustering.h"

class Fast_DOCGPU : public Clustering{
 public:
 Fast_DOCGPU() : Fast_DOCGPU(new std::vector<std::vector<float>*>){};
 Fast_DOCGPU(DataReader* dr) : Fast_DOCGPU(initDataReader(dr)){};
 Fast_DOCGPU(std::vector<std::vector<float>*>* input) : Fast_DOCGPU(input, 0.1, 0.25, 15) {};
 Fast_DOCGPU(float alpha, float beta, float width) : Fast_DOCGPU(new std::vector<std::vector<float>*>, alpha, beta, width){};
	Fast_DOCGPU(std::vector<std::vector<float>*>* input, float alpha, float beta, float width);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster();
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> findKClusters(int k);

	virtual ~Fast_DOCGPU();
	void setAlpha(float value){this->alpha = value;};
	void setBeta(float value){this->beta = value;};
	void setWidth(float value){this->width = value;};
	float getAlpha(){return this->alpha;};
	float getBeta(){return this->beta;};
	float getWidth(){return this->width;};
	
 private:
	float alpha;
	float beta;
	float width;
	unsigned int size;
	unsigned int dim;
	std::vector<std::vector<float>*>* data;
	std::vector<std::vector<float>*>* initDataReader(DataReader* dr);
	float* transformData();
};

#endif
