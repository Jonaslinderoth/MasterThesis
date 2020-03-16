#ifndef MUAPRIORI_H
#define MUAPRIORI_H

#include <iostream>
#include <vector>
#include <boost/dynamic_bitset.hpp>
#include <math.h>

struct Candidate{
	boost::dynamic_bitset<>* item;
	unsigned int support;
	float score;
};



class MuApriori{
 public:
 MuApriori(std::vector<boost::dynamic_bitset<>*>* itemSet, unsigned int minSupp) : MuApriori(itemSet, minSupp, 0.25){};
	MuApriori(std::vector<boost::dynamic_bitset<>*>* itemSet, unsigned int minSupp, float beta);
	std::vector<Candidate>* createInitialCandidates();
	std::vector<Candidate>* createKthCandidates(unsigned int k, std::vector<Candidate>* prevCandidates);
	
	std::vector<Candidate>* findBest(unsigned int numberOfBest);
	float mu(int a, int b){
		return a*pow(((float) 1/this->beta),b);
	};
	float getBeta(){return this->beta;};
	void setBeta(float beta){this->beta = beta;};
	
 private:
	float beta;
	std::vector<boost::dynamic_bitset<>*>* itemSet;
	unsigned int minSupp;
	std::vector<Candidate>* bestCandidates;
};

#endif
