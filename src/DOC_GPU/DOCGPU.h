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
	float* data_d = NULL;
	unsigned int size;
	unsigned int dim;
	std::vector<std::vector<float>*>* data;
	std::vector<std::vector<float>*>* initDataReader(DataReader* dr);

	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findClusterInner(float* data, curandState* randState);
	
};

#endif /* DOCGPU_H_ */
