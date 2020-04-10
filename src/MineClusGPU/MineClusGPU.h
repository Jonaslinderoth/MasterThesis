#ifndef MINECLUSGPU_H
#define MINECLUSGPU_H

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include "../dataReader/DataReader.h"
#include "../Clustering.h"

class MineClusGPU : public Clustering{
 public:
 MineClusGPU() : MineClusGPU(new std::vector<std::vector<float>*>){};
 MineClusGPU(DataReader* dr) : MineClusGPU(initDataReader(dr)){};
 MineClusGPU(std::vector<std::vector<float>*>* input) : MineClusGPU(input, 0.1, 0.25, 15) {};
 MineClusGPU(float alpha, float beta, float width) : MineClusGPU(new std::vector<std::vector<float>*>, alpha, beta, width){};
	MineClusGPU(std::vector<std::vector<float>*>* input, float alpha, float beta, float width);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster();
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> findKClusters(int k);

	virtual ~MineClusGPU();
	float mu(int a, int b){
		return a*pow(((float) 1/this->beta),b);
	};
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
