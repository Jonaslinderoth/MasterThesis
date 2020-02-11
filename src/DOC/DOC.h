#ifndef DOC_h
#define DOC_h
#include <vector>
#include <math.h>
#include <random>
#include <iostream>
#include "../dataReader/DataReader.h"

class DOC{
 public:
	DOC();
	DOC(DataReader* dr);
 DOC(float alpha, float beta, float width) : DOC(new std::vector<std::vector<float>*>, alpha, beta, width){};
	DOC(std::vector<std::vector<float>*>* input);
	DOC(std::vector<std::vector<float>*>* input, float alpha, float beta, float width);

	bool addPoint(std::vector<float>* point);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster();
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> findKClusters(int k);
	
	int size();
	std::vector<std::vector<float>*>* pickRandom(int n);
	std::vector<bool>* findDimensions(std::vector<float>* centroid, std::vector<std::vector<float>*>* points, float width);

	float mu(int a, int b){
		return a*pow(((float) 1/this->beta),b);
	};

	void setSeed(int s){
		this->gen.seed(s);
	}

	void setAlpha(float value){this->alpha = value;};
	void setBeta(float value){this->beta = value;};
	void setWidth(float value){this->width = value;};

private:
	float alpha;
	float beta;
	float width;
	std::vector<std::vector<float>*>* data;
	std::mt19937 gen;
	size_t seed = std::random_device()();
	
	std::vector<int> randInt(int upper, int n){
		return randInt(0,upper, n);
	};
	std::vector<int> randInt(int lower, int upper, int n){
		std::vector<int> res = std::vector<int>();
				
		//std::random_device rd;  //Will be used to obtain a seed for the random number engine
		//std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_int_distribution<> dis(lower, upper); //inclusive

		for(int i = 0; i < n; i++){
			auto a = dis(gen);

			res.push_back(a);
		}
		return res;
	};
	
	std::vector<std::vector<float>*>* initDataReader(DataReader* dr);

};


#endif
