/*
 * HyperCube.cpp
 *
 *  Created on: 04/02/2020
 *      Author: jonas
 */

#include <src/DOC/HyperCube.h>
#include <algorithm>    // std::min




HyperCube::HyperCube(std::vector<float> centroid, float width,
		std::vector<bool> dimmensions) {
	this->centroid = centroid;
	this->width = width;
	this->setDimmensions(dimmensions);
}

HyperCube::~HyperCube() {
	// TODO Auto-generated destructor stub
}

bool HyperCube::pointContained(std::vector<float> point) {
	for(int i = 0; i < centroid.size(); i++){
		if(i < dimmensions.size() && dimmensions[i]){
			float a = this->centroid.at(i)-this->width;
			float b = this->centroid.at(i)+this->width;
			float min = std::min(a,b);
			float max = std::max(a,b);
			if(not (min < point.at(i) && max > point.at(i))){
				return false;
			}
		}
	}
	return true;
}
