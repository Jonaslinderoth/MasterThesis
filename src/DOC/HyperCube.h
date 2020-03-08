/*
 * HyperCube.h
 *
 *  Created on: 04/02/2020
 *      Author: jonas
 */

#ifndef HYPERCUBE_H_
#define HYPERCUBE_H_
#include <vector>

class HyperCube {
public:

	HyperCube(std::vector<float>* centroid, float width, std::vector<bool>* dimmensions);
	HyperCube(std::vector<float>* centroid, float width) : HyperCube(centroid, width, new std::vector<bool>){};
	virtual ~HyperCube();
	bool pointContained(std::vector<float>* point);

	const std::vector<float>* getCentroid() const {
		return centroid;
	}

	void setCentroid(std::vector<float>* centroid) {
		this->centroid = centroid;
	}

	const std::vector<bool>* getDimmensions() const {
		return dimmensions;
	}

	void setDimmensions(std::vector<bool>* dimmensions) {
		this->dimmensions = dimmensions;
		while(this->dimmensions->size() < this->centroid->size()){
			this->dimmensions->push_back(false);
		}

	}


	const bool getDimmension(int i) {
		return this->dimmensions->at(i);
	}

	void setDimmension(int dimmension, bool value) {
		this->dimmensions->at(dimmension) = value;
	}

	float getWidth() const {
		return width;
	}

	void setWidth(float width) {
		this->width = width;
	}

private:
	float width;
	std::vector<float>* centroid;
	std::vector<bool>* dimmensions;

};

#endif /* HYPERCUBE_H_ */
