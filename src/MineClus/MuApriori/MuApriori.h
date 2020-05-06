#ifndef MUAPRIORI_H
#define MUAPRIORI_H

#include <iostream>
#include <vector>
#include <queue>
#include <boost/dynamic_bitset.hpp>
#include <math.h>

struct Candidate{
	boost::dynamic_bitset<> item;
	unsigned int support;
	float score;
};

struct OutputCandidate:Candidate{
	unsigned int centroidNr;
	OutputCandidate(){
	};
	OutputCandidate(const Candidate& b) {
		this->item = b.item;
		this->support = b.support;
		this->score = b.score;
	};
	
};





class MuApriori{
 public:
 MuApriori(std::vector<boost::dynamic_bitset<>>* itemSet, unsigned int minSupp) : MuApriori(itemSet, minSupp, 0.25){};
	MuApriori(std::vector<boost::dynamic_bitset<>>* itemSet, unsigned int minSupp, float beta);
	std::vector<Candidate*>* createInitialCandidates();
	std::vector<Candidate*>* createKthCandidates(unsigned int k, std::vector<Candidate*>* prevCandidates);
	
	void findBest(unsigned int numberOfBest);
	float mu(int a, int b){
		return a*pow(((float) 1/this->beta),b);
	};
	float getBeta(){return this->beta;};
	void setBeta(float beta){this->beta = beta;};
	
	void setItemSet(std::vector<boost::dynamic_bitset<>>* itemSet){
		this->centroidNr++;
		this->itemSet = itemSet;
	};

	OutputCandidate* getBest(){
		return bestCandidate;
	};
	
 private:
	float beta;
	std::vector<boost::dynamic_bitset<>>* itemSet;
	unsigned int minSupp;
	unsigned int numberOfCandidates = 1;
	unsigned int centroidNr;
	struct CustomCompare{
		bool operator()(const OutputCandidate & a, const OutputCandidate & b){
			return (a.score) > (b.score);
		};		
	};

	
	OutputCandidate* bestCandidate = nullptr;


};

#endif
