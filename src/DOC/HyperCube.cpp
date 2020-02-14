/*
 * HyperCube.cpp
 *
 *  Created on: 04/02/2020
 *      Author: jonas
 */

#include <src/DOC/HyperCube.h>
#include <algorithm>    // std::min




HyperCube::HyperCube(std::vector<float>* centroid, float width,
		std::vector<bool>* dimmensions) {
	this->centroid = centroid;
	this->width = width;
	this->setDimmensions(dimmensions);
}

HyperCube::~HyperCube() {
	// TODO Auto-generated destructor stub
}

bool HyperCube::pointContained(std::vector<float>* point) {
	bool res = true;
	for(int i = 0; i < centroid->size(); i++){
		auto r = (!this->dimmensions->at(i))|| abs(this->centroid->at(i)-point->at(i)) < this->width;
		if(!r) return false;
	}
	return true;
}
