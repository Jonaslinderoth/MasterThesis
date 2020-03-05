/*
 * DOCGPU.h
 *
 *  Created on: Feb 14, 2020
 *      Author: mikkel
 */

#ifndef DOCGPU_H_
#define DOCGPU_H_

#include <stdio.h>
#include <stdlib.h>
#include <random>
//# include <cuda.h>
# include <curand.h>
#include "../dataReader/DataReader.h"
#include "../Clustering.h"
#include "DOCGPU_Kernels.h"
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>


class DOCGPU : public Clustering{
public:
 DOCGPU() : DOCGPU(new std::vector<std::vector<float>*>){};
 DOCGPU(DataReader* dr) : DOCGPU(initDataReader(dr)){};
 DOCGPU(std::vector<std::vector<float>*>* input) : DOCGPU(input, 0.1, 0.25, 15) {};
 DOCGPU(float alpha, float beta, float width) : DOCGPU(new std::vector<std::vector<float>*>, alpha, beta, width){};
	DOCGPU(std::vector<std::vector<float>*>* input, float alpha, float beta, float width);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster();
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> findKClusters(int k);

	virtual ~DOCGPU();
	void setAlpha(float value){this->alpha = value;};
	void setBeta(float value){this->beta = value;};
	void setWidth(float value){this->width = value;};
	unsigned int getSize(){return this->size;};
	unsigned int dimension(){return this->dim;};


 private:
	float alpha;
	float beta;
	float width;
	unsigned int size;
	unsigned int dim;
	std::vector<std::vector<float>*>* data;
	std::vector<std::vector<float>*>* initDataReader(DataReader* dr);
	float* transformData();

	bool isAllocated = false;
	size_t allocated_size_of_data;
	size_t allocated_size_of_samples;
	size_t allocated_size_of_centroids;
	size_t allocated_size_of_findDim;
	size_t allocated_size_of_findDim_count;
	size_t allocated_size_of_pointsContained;
	size_t allocated_size_of_pointsContained_count;
	size_t allocated_size_of_score;
	size_t allocated_size_of_index;
	size_t allocated_size_of_randomStates;
	size_t allocated_size_of_bestDims;

	size_t getAllocated(size_t value){
		if(!isAllocated){
			return SIZE_MAX;
		}else{
			return value;
		}
	}
	
	size_t get_size_of_data(){ return getAllocated(this->allocated_size_of_data);}
	size_t get_size_of_samples(){ return getAllocated(this->allocated_size_of_samples);}
	size_t get_size_of_centroids(){ return getAllocated(this->allocated_size_of_centroids);}
	size_t get_size_of_findDim(){ return getAllocated(this->allocated_size_of_findDim);}
	size_t get_size_of_pointsContained(){ return getAllocated(this->allocated_size_of_pointsContained);}
	size_t get_size_of_pointsContained_count(){ return getAllocated(this->allocated_size_of_pointsContained_count);}
	size_t get_size_of_score(){ return getAllocated(this->allocated_size_of_score);}
	size_t get_size_of_index(){ return getAllocated(this->allocated_size_of_index);}
	size_t get_size_of_randomStates(){ return getAllocated(this->allocated_size_of_randomStates);}
	size_t get_size_of_bestDims(){ return getAllocated(this->allocated_size_of_bestDims);}
	size_t get_size_of_findDim_count(){ return getAllocated(this->allocated_size_of_findDim_count);}



	
};

#endif /* DOCGPU_H_ */
