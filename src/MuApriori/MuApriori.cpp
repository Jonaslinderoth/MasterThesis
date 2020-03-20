#include "MuApriori.h"
#include <math.h>

MuApriori::MuApriori(std::vector<boost::dynamic_bitset<>*>* itemSet, unsigned int minSupp, float beta){
	this->itemSet = itemSet;
	this->minSupp = minSupp;
	this->beta = beta;
	this->bestCandidates = new std::priority_queue<Candidate, std::vector<Candidate>,  CustomCompare>;
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
			result->at(index).support++;			
			index = this->itemSet->at(i)->find_next(index);
		}
	}
	unsigned int i = 0;
	while(i< result->size()){
		if(result->at(i).support < this->minSupp){
			result->erase(result->begin()+i); // todo better delete
		}else{
			result->at(i).score = this->mu(result->at(i).support, result->at(i).item->count());
			if(this->bestCandidates->size() < this->numberOfCandidates){
				this->bestCandidates->push(result->at(i));
			}else{
				if(this->bestCandidates->top().score < result->at(i).score){
					this->bestCandidates->pop();
					this->bestCandidates->push(result->at(i));
				}
			}
			
			i++;
		}
	}

	return result;
};


std::vector<Candidate>* MuApriori::createKthCandidates(unsigned int k, std::vector<Candidate>* prevCandidates){
	size_t dim = this->itemSet->at(0)->size();
	std::vector<Candidate>* result = new std::vector<Candidate>;

	// Merge the candidates
	for(unsigned int i = 0; i < prevCandidates->size(); i++){
		for(unsigned int j = 0; j < i; j++){
			boost::dynamic_bitset<> intersection = boost::dynamic_bitset<>(*(prevCandidates->at(i).item));
			intersection &= *(prevCandidates->at(j).item);
			size_t intersection_count = intersection.count();
			if(intersection_count >= k-2){ 
				boost::dynamic_bitset<>* union2 = new boost::dynamic_bitset<>(*(prevCandidates->at(i).item));
				*union2 |= *(prevCandidates->at(j).item);
				size_t union_count = union2->count();
				Candidate c;
				c.item = union2;
				c.support = 0;
				c.score = 0;
				result->push_back(c);
			}
		}
	}

	// remove dublicates
	std::sort( result->begin(), result->end(), [](const Candidate & a, const Candidate & b) -> bool{
			return *(a.item) > *(b.item);
		});
	result->erase( std::unique( result->begin(), result->end(),[](const Candidate & a, const Candidate & b) -> bool{
			return *(a.item) == *(b.item);
		}), result->end() );

	
	// Count support
	for(unsigned int i = 0; i < this->itemSet->size(); i++){
		for(unsigned int j = 0; j < result->size(); j++){
			//Test if the j'th candidate is a subset of the i'th point
			boost::dynamic_bitset<>* union2 = new boost::dynamic_bitset<>(*(this->itemSet->at(i)));
			size_t u_count1 = union2->count();
			*union2 |= *(result->at(j).item);
			size_t u_count2 = union2->count();
			if(u_count1 == u_count2){
				result->at(j).support++;
			}
		}
	}


	// calculate the score, and store the best. 
	unsigned int i = 0;
	while(i<result->size()){
		if(result->at(i).support < this->minSupp){
			result->erase(result->begin()+i); // todo better delete
		}else{
			result->at(i).score = this->mu(result->at(i).support, result->at(i).item->count());
			if(this->bestCandidates->size() < this->numberOfCandidates){
				this->bestCandidates->push(result->at(i));
			}else{
				if(this->bestCandidates->top().score < result->at(i).score){
					this->bestCandidates->pop();
					this->bestCandidates->push(result->at(i));
				}
			}
			i++;
		}
	}

	return result;
};


std::vector<Candidate>* MuApriori::findBest(unsigned int numberOfBest){
	this->numberOfCandidates = numberOfBest;
	std::vector<Candidate>* result = this->createInitialCandidates();
	unsigned int dim = result->at(0).item->size();
	unsigned int k = 1;
	if(numberOfBest == 1){
		while(result->size() > 0 && k < dim){
			k ++;	
			result = this->createKthCandidates(k, result);
		}
	}else{
		throw std::runtime_error("Not Implemented");	
	}
	auto output = new std::vector<Candidate>;
	for(int i = 0; i < this->bestCandidates->size(); i++){
		output->push_back(this->bestCandidates->top());
		this->bestCandidates->pop();
	}

	 std::reverse(output->begin(),output->end());
	 
	return output;
};
