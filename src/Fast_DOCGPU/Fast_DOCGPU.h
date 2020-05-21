#ifndef FAST_DOCGPU_H
#define FAST_DOCGPU_H
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include "../dataReader/DataReader.h"
#include "../Clustering.h"
#include "../DOC_GPU/DOCGPU_Kernels.h"
#include "../Fast_DOCGPU/whatDataInCentroid.h"

class Fast_DOCGPU : public Clustering{
 public:
 Fast_DOCGPU() : Fast_DOCGPU(new std::vector<std::vector<float>*>){};
 Fast_DOCGPU(DataReader* dr) : Fast_DOCGPU(initDataReader(dr)){};
 Fast_DOCGPU(std::vector<std::vector<float>*>* input) : Fast_DOCGPU(input, 0.1, 0.25, 15) {};
 Fast_DOCGPU(float alpha, float beta, float width) : Fast_DOCGPU(alpha,beta, width, 0){}
 Fast_DOCGPU(float alpha, float beta, float width, unsigned int d0) : Fast_DOCGPU(new std::vector<std::vector<float>*>, alpha, beta, width){};
 Fast_DOCGPU(std::vector<std::vector<float>*>* input, float alpha, float beta, float width) : Fast_DOCGPU(input, alpha, beta, width,0){};
	Fast_DOCGPU(std::vector<std::vector<float>*>* input, float alpha, float beta, float width, unsigned int d0);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster();
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> findKClusters(int k);

	virtual ~Fast_DOCGPU();
	void setAlpha(float value){this->alpha = value;};
	void setBeta(float value){this->beta = value;};
	void setWidth(float value){this->width = value;};
	float getAlpha(){return this->alpha;};
	float getBeta(){return this->beta;};
	float getWidth(){return this->width;};
	void setd0(unsigned int value){this->d0 = value;};
	unsigned int getd0(){return this->d0;};
	void setFindDimVersion(findDimVersion a){
		this->findDimKernelVersion = a;
	}
	void setContainedVersion(containedType a){
		this->containedVersion = a;
	}
 private:
	float alpha;
	float beta;
	float width;
	unsigned int size;
	unsigned int dim;
	unsigned int d0;
	std::vector<std::vector<float>*>* data;
	std::vector<std::vector<float>*>* initDataReader(DataReader* dr);
	float* transformData();
	findDimVersion findDimKernelVersion = naiveFindDim;
	containedType containedVersion = NaiveContained;
};

#endif
