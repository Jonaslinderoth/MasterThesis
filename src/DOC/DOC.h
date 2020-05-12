#ifndef DOC_h
#define DOC_h
#include <vector>
#include <math.h>
#include <random>
#include <iostream>
#include "../dataReader/DataReader.h"
#include "../Clustering.h"

class DOC : public Clustering{
 public:
	DOC();
	DOC(DataReader* dr);
 DOC(float alpha, float beta, float width) : DOC(new std::vector<std::vector<float>*>, alpha, beta, width){};
	DOC(std::vector<std::vector<float>*>* input);
	DOC(std::vector<std::vector<float>*>* input, float alpha, float beta, float width);

	bool addPoint(std::vector<float>* point);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster() override;
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> findKClusters(int k) override;
	
	int size();
	std::vector<std::vector<float>*>* pickRandom(int n);
	std::vector<bool>* findDimensions(std::vector<float>* centroid, std::vector<std::vector<float>*>* points, float width);

	float mu(int a, int b){
		return log(a)+log((float) 1/this->beta)*b;
	};
	void setNumberOfSamples(unsigned int m){this->m = m;};
	void setAlpha(float value){this->alpha = value;};
	void setBeta(float value){this->beta = value;};
	void setWidth(float value){this->width = value;};
	virtual ~DOC(){};
private:
	unsigned int m = 0;
	float alpha;
	float beta;
	float width;
	std::vector<std::vector<float>*>* data;
	std::vector<std::vector<float>*>* initDataReader(DataReader* dr);

};


#endif
