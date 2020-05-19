#ifndef FAST_DOCGPUNIFIED_H
#define FAST_DOCGPUNIFIED_H
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include "../dataReader/DataReader.h"
#include "../Clustering.h"
#include "../DOC_GPU/DOCGPU_Kernels.h"

class Fast_DOCGPUnified : public Clustering{
 public:
 Fast_DOCGPUnified() : Fast_DOCGPUnified(new std::vector<std::vector<float>*>){};
 Fast_DOCGPUnified(DataReader* dr) : Fast_DOCGPUnified(initDataReader(dr)){};
 Fast_DOCGPUnified(std::vector<std::vector<float>*>* input) : Fast_DOCGPUnified(input, 0.1, 0.25, 15) {};
 Fast_DOCGPUnified(float alpha, float beta, float width) : Fast_DOCGPUnified(alpha,beta, width, 0){}
 Fast_DOCGPUnified(float alpha, float beta, float width, unsigned int d0) : Fast_DOCGPUnified(new std::vector<std::vector<float>*>, alpha, beta, width){};
 Fast_DOCGPUnified(std::vector<std::vector<float>*>* input, float alpha, float beta, float width) : Fast_DOCGPUnified(input, alpha, beta, width,0){};
	Fast_DOCGPUnified(std::vector<std::vector<float>*>* input, float alpha, float beta, float width, unsigned int d0);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster();
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> findKClusters(int k);

	virtual ~Fast_DOCGPUnified();
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
};

#endif
