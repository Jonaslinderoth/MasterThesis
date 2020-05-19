#ifndef MINECLUSGPUNIFIED_H
#define MINECLUSGPUNIFIED_H

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include "../dataReader/DataReader.h"
#include "../Clustering.h"
#include "FindDublicates.h"
#include "CountSupport.h"

class MineClusGPUnified : public Clustering{
 public:
 MineClusGPUnified() : MineClusGPUnified(new std::vector<std::vector<float>*>){};
 MineClusGPUnified(DataReader* dr) : MineClusGPUnified(initDataReader(dr)){};
 MineClusGPUnified(std::vector<std::vector<float>*>* input) : MineClusGPUnified(input, 0.1, 0.25, 15) {};
 MineClusGPUnified(float alpha, float beta, float width) : MineClusGPUnified(new std::vector<std::vector<float>*>, alpha, beta, width){};
	MineClusGPUnified(std::vector<std::vector<float>*>* input, float alpha, float beta, float width);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster();
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> findKClusters(int k);

	virtual ~MineClusGPUnified();
	float mu(int a, int b){
		return a*pow(((float) 1/this->beta),b);
	};
	void setAlpha(float value){this->alpha = value;};
	void setBeta(float value){this->beta = value;};
	void setWidth(float value){this->width = value;};
	float getAlpha(){return this->alpha;};
	float getBeta(){return this->beta;};
	float getWidth(){return this->width;};
	bool isConcurentVersion(){return this->concurentVersion;};
	void setConcurentVersion(bool value){this->concurentVersion = value;};
	void setDuplicatesVersion(dublicatesType a){
		this->duplicateKernelVerison = a; 
	}
	void setCountSupportVersion(countSupportType a){
		this->countSupportKernelVersion = a;
	}
	
 private:
	float alpha;
	float beta;
	float width;
	unsigned int size;
	unsigned int dim;
	bool concurentVersion = true;
	std::vector<std::vector<float>*>* data;
	std::vector<std::vector<float>*>* initDataReader(DataReader* dr);
	float* transformData();
	dublicatesType duplicateKernelVerison = Hash;
	countSupportType countSupportKernelVersion = NaiveCount;
};

#endif
