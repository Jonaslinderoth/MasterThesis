#include "MuApriori.h"


MuApriori::MuApriori(std::vector<boost::dynamic_bitset<>*>* itemSet, unsigned int minSupp, float beta){
	throw std::runtime_error("Not Implemented");	
};

std::vector<Candidate>* MuApriori::createInitialCandidates(){
	throw std::runtime_error("Not Implemented");	
};


void MuApriori::createKthCandidates(unsigned int k, std::vector<Candidate>* prevCandidates){
	throw std::runtime_error("Not Implemented");		
};


std::vector<Candidate>* MuApriori::findBest(unsigned int numberOfBest){
	if(numberOfBest == 1){
		
	}else{
		throw std::runtime_error("Not Implemented");	
	}
};
