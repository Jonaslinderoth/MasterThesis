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

	std::vector<OutputCandidate*>* getBest(){
		auto output = new std::vector<OutputCandidate*>;
		auto numberOfCandidates = this->bestCandidates->size();
		for(int i = 0; i < numberOfCandidates; i++){
			OutputCandidate* c = new OutputCandidate;
			c->item = (this->bestCandidates->top().item);
			c->support = this->bestCandidates->top().support;
			c->score = this->bestCandidates->top().score;
			c->centroidNr = this->bestCandidates->top().centroidNr;
			output->push_back(c);
			this->bestCandidates->pop();
		}
		std::reverse(output->begin(),output->end());
	 
		return output;
	}
	
 private:
	float beta;
	std::vector<boost::dynamic_bitset<>>* itemSet;
	unsigned int minSupp;
	unsigned int numberOfCandidates;
	unsigned int centroidNr;
	struct CustomCompare{
		bool operator()(const OutputCandidate & a, const OutputCandidate & b){
			return (a.score) > (b.score);
		};		
	};

	
	std::priority_queue<OutputCandidate, std::vector<OutputCandidate>,  CustomCompare>* bestCandidates;


};

#endif
