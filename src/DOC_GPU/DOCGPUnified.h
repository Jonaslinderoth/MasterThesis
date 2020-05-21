/*
 * DOCGPU.h
 *
 *  Created on: Feb 14, 2020
 *      Author: mikkel
 */

#ifndef DOCGPUNIFIED_H_
#define DOCGPUNIFIED_H_

#include <stdio.h>
#include <stdlib.h>
#include <random>
//# include <cuda.h>
#include <curand.h>
#include "../dataReader/DataReader.h"
#include "../Clustering.h"
#include "DOCGPU_Kernels.h"
#include "pointsContainedDevice.h"
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>


class DOCGPUnified : public Clustering{
public:
 DOCGPUnified() : DOCGPUnified(new std::vector<std::vector<float>*>){};
 DOCGPUnified(DataReader* dr) : DOCGPUnified(initDataReader(dr)){};
 DOCGPUnified(std::vector<std::vector<float>*>* input) : DOCGPUnified(input, 0.1, 0.25, 15) {};
 DOCGPUnified(float alpha, float beta, float width) : DOCGPUnified(new std::vector<std::vector<float>*>, alpha, beta, width){};
	DOCGPUnified(std::vector<std::vector<float>*>* input, float alpha, float beta, float width);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster();
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> findKClusters(int k);

	virtual ~DOCGPUnified();
	void setAlpha(float value){this->alpha = value;};
	void setBeta(float value){this->beta = value;};
	void setWidth(float value){this->width = value;};
	unsigned int getSize(){return this->size;};
	unsigned int dimension(){return this->dim;};
	void setNumberOfSamples(unsigned int m){this->m = m;};
	void setFindDimVersion(findDimVersion a){
		this->findDimKernelVersion = a;
	}
	void setPointsContainedVersion(pointContainedType a){
		this->pointsContainedVersion = a;
	}

 private:
	unsigned int m = 0;
	float alpha;
	float beta;
	float width;
	unsigned int size;
	unsigned int dim;
	std::vector<std::vector<float>*>* data;
	std::vector<std::vector<float>*>* initDataReader(DataReader* dr);
	float* transformData();
	findDimVersion findDimKernelVersion = naiveFindDim;
	pointContainedType pointsContainedVersion = pointContainedNaive;



	
};

#endif /* DOCGPUnified_H_ */
