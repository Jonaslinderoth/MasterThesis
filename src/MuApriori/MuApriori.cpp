#include "MuApriori.h"


MuApriori::MuApriori(std::vector<boost::dynamic_bitset<>*>* itemSet, unsigned int minSupp, float beta){
	this->itemSet = itemSet;
	this->minSupp = minSupp;
	this->beta = beta;
};

std::vector<Candidate>* MuApriori::createInitialCandidates(){
	throw std::runtime_error("Not Implemented");	
};


void MuApriori::createKthCandidates(unsigned int k, std::vector<Candidate>* prevCandidates){
	throw std::runtime_error("Not Implemented");		
};


std::vector<Candidate>* MuApriori::findBest(unsigned int numberOfBest){
	if(numberOfBest == 1){
		throw std::runtime_error("Not Implemented");			
	}else{
		throw std::runtime_error("Not Implemented");	
	}
};
