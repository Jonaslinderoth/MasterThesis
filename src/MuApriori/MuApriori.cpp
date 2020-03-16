#include "MuApriori.h"


MuApriori::MuApriori(std::vector<boost::dynamic_bitset<>*>* itemSet, unsigned int minSupp, float beta){
	this->itemSet = itemSet;
	this->minSupp = minSupp;
	this->beta = beta;
	this->bestCandidates = new std::vector<Candidate>;
};

std::vector<Candidate>* MuApriori::createInitialCandidates(){
	std::vector<Candidate>* result = new std::vector<Candidate>;
	size_t dim = this->itemSet->at(0)->size();
	
	for(unsigned int i = 0; i<dim; i++){
		Candidate c;
		c.item = new boost::dynamic_bitset<>(dim,0);
		c.item->operator[](i) = true;
		c.support = 0;
		c.score = 0;
		result->push_back(c);
	}


	for(unsigned int i = 0; i<this->itemSet->size(); i++){
		size_t index = this->itemSet->at(i)->find_first();
		while(index != boost::dynamic_bitset<>::npos){
			std::cout << index << std::endl;
			result->at(index).support++;			
			index = this->itemSet->at(i)->find_next(index);
		}
	}

	

	unsigned int i = 0;
	while(i< result->size()){
		std::cout << i << " " << *(result->at(i).item) << std::endl;
		if(result->at(i).support < this->minSupp){
			result->erase(result->begin()+i); // todo better delete
		}else{
			result->at(i).score = this->mu(result->at(i).support, result->at(i).item->count());
			if(this->bestCandidates->size() == 0){
				this->bestCandidates->push_back(result->at(i));				
			}else if(result->at(i).score > this->bestCandidates->at(0).score){
				this->bestCandidates->at(0) = result->at(i);
			}
			i++;
		}
	}

	return result;
};


std::vector<Candidate>* MuApriori::createKthCandidates(unsigned int k, std::vector<Candidate>* prevCandidates){
	std::cout << "finding the " << k << "'th round of candidates" << std::endl;
	size_t dim = this->itemSet->at(0)->size();
	std::vector<Candidate>* result = new std::vector<Candidate>;
	for(unsigned int i = 0; i < prevCandidates->size(); i++){
		for(unsigned int j = 0; j < i; j++){
			boost::dynamic_bitset<> intersection = boost::dynamic_bitset<>(*(prevCandidates->at(i).item));
			intersection &= *(prevCandidates->at(j).item);
			size_t intersection_count = intersection.count();
			if(intersection_count >= k-2 || intersection_count >= 0){
				boost::dynamic_bitset<>* union2 = new boost::dynamic_bitset<>(*(prevCandidates->at(i).item));
				*union2 |= *(prevCandidates->at(j).item);
				size_t union_count = union2->count();
				std::cout << *union2 << " Candidate candidate" << std::endl;
				// if the score can be better than the current best
				if(this->mu(prevCandidates->at(i).item->count()+prevCandidates->at(j).item->count(), union_count) > this->bestCandidates->at(0).score){
					Candidate c;
					
					c.item = union2;
					std::cout << "new candidate " << *union2 << std::endl;
					c.support = 0;
					c.score = 0;
					result->push_back(c);
				}
			}
		}
	}

	
	for(unsigned int i = 0; i < this->itemSet->size(); i++){
		for(unsigned int j = 0; j < result->size(); j++){
			//Test if the j'th candidate is a subset of the i'th point
			boost::dynamic_bitset<>* union2 = new boost::dynamic_bitset<>(*(this->itemSet->at(i)));
			size_t u_count1 = union2->count();
			*union2 |= *(result->at(j).item);
			size_t u_count2 = union2->count();
			if(u_count1 == u_count2){
				std::cout << *union2 << " is a subset of " << *(this->itemSet->at(i)) << std::endl;
				result->at(j).support++;
			}
		}
	}


	unsigned int i = 0;
	while(i<result->size()){
		std::cout << i << std::endl;
		result->at(i).score = this->mu(result->at(i).support, result->at(i).item->count());
		if(result->at(i).support < this->minSupp || result->at(i).score <= this->bestCandidates->at(0).score){
			std::cout << "deleted: " << *(result->at(i).item) << std::endl;
			result->erase(result->begin()+i); // todo better delete
		}else{
			if(this->bestCandidates->size() == 0){
				std::cout << "new best score" << result->at(i).score << std::endl;
				this->bestCandidates->push_back(result->at(i));				
			}else if(result->at(i).score > this->bestCandidates->at(0).score){
				std::cout << "new best score" << result->at(i).score << std::endl;
				this->bestCandidates->at(0) = result->at(i);
			}else{
				std::cout << result->at(i).score << "not new best" << std::endl;
			}
			i++;
		}
	}

	return result;
};


std::vector<Candidate>* MuApriori::findBest(unsigned int numberOfBest){
	std::vector<Candidate>* result = this->createInitialCandidates();
	unsigned int k = 1;
	if(numberOfBest == 1){
		while(result->size() > 0){
			std::cout << "best is " << *(this->bestCandidates->at(0).item) << " with support " << this->bestCandidates->at(0).support << " and score " << this->bestCandidates->at(0).score << std::endl;
			k ++;	
			result = this->createKthCandidates(k, result);
		}
		std::cout << "best is " << *(this->bestCandidates->at(0).item) << " with support " << this->bestCandidates->at(0).support << " and score " << this->bestCandidates->at(0).score << std::endl;
		
	}else{
		throw std::runtime_error("Not Implemented");	
	}
	return this->bestCandidates;
};
