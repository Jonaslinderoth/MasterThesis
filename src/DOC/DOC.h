#ifndef DOC_h
#define DOC_h
#include <vector>
#include <math.h>
#include <random>
#include <iostream>
#include "../Cluster.h"

class DOC : public Cluster{
public:
	DOC();
	DOC(std::vector<std::vector<float>*>* input);
	DOC(std::vector<std::vector<float>*>* input, float alpha, float beta, float width);

	bool addPoint(std::vector<float>* point);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster();
	int size();
	std::vector<std::vector<float>*>* pickRandom(int n);
	std::vector<bool>* findDimensions(std::vector<float>* centroid, std::vector<std::vector<float>*>* points, float width);

	float mu(int a, int b){
		return a*pow(((float) 1/this->beta),b);
	};

	const std::vector<int>& getRandomStub() const {
		return randomStub;
	};

	void setRandomStub(const std::vector<int>& randomStub) {
		this->randomStub = randomStub;
	};

	void setRandom(bool value){
			this->trueRandom = value;
	};

	float findCandidateCluster(std::vector<float>* p, std::vector<std::vector<float>*>* X,
				  std::vector<std::vector<float>*>* resC, std::vector<bool>* resD, int d, float maxValue);
	
private:
	float alpha;
	float beta;
	float width;
	bool trueRandom = true;
	std::vector<std::vector<float>*>* data;
	std::vector<int> randomStub;

	std::vector<int> randInt(int upper, int n){
		return randInt(0,upper, n);
	};
	std::vector<int> randInt(int lower, int upper, int n){
		std::vector<int> res = std::vector<int>();
		if (this->trueRandom){
			std::random_device rd;  //Will be used to obtain a seed for the random number engine
			std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
			std::uniform_int_distribution<> dis(lower, upper); //inclusive
			for(int i = 0; i < n; i++){
				res.push_back(dis(gen));
			}
		}else{
			for(int i = 0; i < n; i++){
					res.push_back(this->randomStub.at(i%(this->randomStub.size())));
			}
		}
		return res;
	};

};


#endif
