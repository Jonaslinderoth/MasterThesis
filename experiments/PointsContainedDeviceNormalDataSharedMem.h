/*
 * PointsContainedDeviceNormalData.h
 *
 *  Created on: May 19, 2020
 *      Author: mikkel
 */

#ifndef POINTSCONTAINEDDEVICENORMALDATASMEM_H_
#define POINTSCONTAINEDDEVICENORMALDATASMEM_H_

#include "Experiment.h"
#include <random>
#include <vector>
class PointsContainedDeviceNormalDataSharedMem: public Experiment{
 public:
	PointsContainedDeviceNormalDataSharedMem(std::string name, std::string dir): Experiment(0,name, dir, "number of points, dim, version, block size, shared memory size, time"){}
	void start() override;
 private:
	std::vector<std::vector<float>*>* pickRandomPointFromData(std::vector<std::vector<float>*>* data, unsigned int size);
	std::vector<int> randIntVec(unsigned int lower,unsigned int upper,unsigned int n);
	std::vector<bool>* findDimensionsEx(std::vector<float>* centroid,std::vector<std::vector<float>* >* points,float width);
	std::mt19937 gen;
};





#endif /* POINTSCONTAINEDDEVICENORMALDATA_H_ */
