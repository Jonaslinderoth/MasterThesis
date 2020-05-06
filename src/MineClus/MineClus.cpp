#include "MineClus.h"
#include <boost/dynamic_bitset.hpp>
#include <stdexcept>
#include <math.h>
#include <algorithm>
#include "../DOC/HyperCube.h"
// #include "../MuApriori/MuApriori.h"



MineClus::MineClus(std::vector<std::vector<float>*>* input, float alpha, float beta, float width, unsigned int d0){
	this->data = input;
	this->alpha = alpha;
	this->width = width;
	this->beta = beta;
	if(d0 == 0){
		if(input->size() > 0){
			this->d0 = input->at(0)->size();		
		}else{
			this->d0 = 1;
		}

	}else{
		this->d0 = d0;
	}

}


std::vector<OutputCandidate*>* MineClus::findClusterCandidates(){
		float d = this->data->at(0)->size();
	unsigned int min_supp = this->alpha*this->data->size();
	unsigned int outer = (float)2/this->alpha;
	this->medoids = this->pickRandom(outer);
	MuApriori* muApriori;
	std::vector<OutputCandidate*>* output = new std::vector<OutputCandidate*>;
	for(int i = 0; i < outer; i++){
		// compute itemSet
		std::vector<boost::dynamic_bitset<>>* itemSet = this->findDimensions(this->medoids->at(i),
							  this->data, this->width);
		// call apriori
		muApriori = new MuApriori(itemSet, min_supp, this->beta);
		muApriori->findBest(1);
		auto temp = muApriori->getBest();
		if(temp != nullptr){
			temp->centroidNr = i;
			output->push_back(temp);
		}
		delete muApriori;
	}
	
	return output;
}

std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> MineClus::findCluster(){

	auto clusterCandidates = this->findClusterCandidates();
	OutputCandidate* best = nullptr;
	for(unsigned int i = 0; i <clusterCandidates->size(); i++){
		if(clusterCandidates->at(i) != nullptr && (best == nullptr || best->score < clusterCandidates->at(i)->score)){
			best = clusterCandidates->at(i);
		}
	}

	auto subspace = new std::vector<bool>;
	for(int i = 0; i < best->item.size(); i++){
		subspace->push_back(best->item[i]);
	}

	auto cube = HyperCube((this->medoids->at(best->centroidNr)),this->width, subspace);
	std::vector<std::vector<float>*>* resC = new std::vector<std::vector<float>*>;

	for(int l = 0; l < this->data->size(); l++){
		auto point = this->data->at(l);
		if (cube.pointContained(point)){
			resC->push_back(point);
		}
	}

	for(unsigned int i = 0; i < clusterCandidates->size(); i++){
		delete clusterCandidates->at(i);
	}
	delete clusterCandidates;
	auto result = std::make_pair(resC, subspace);
	return result;

};



bool MineClus::isDisjoint(OutputCandidate* lhs, OutputCandidate* rhs){
	auto intersection = lhs->item & rhs->item;
	auto dim = this->data->at(0)->size();
	for(unsigned int i = 0; i < dim; i++){
		if(intersection[i] && abs(this->medoids->at(lhs->centroidNr)->at(i) -this->medoids->at(rhs->centroidNr)->at(i)) >= 2*this->width){
			return true;
		};
	}
	return false;
};



std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> MineClus::findKClusters(int k){
	auto res = std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>>();
	while(res.size() < k){
		if(this->data->size() <= 0) {break;}
		auto clusterCandidates = this->findClusterCandidates();
		std::vector<OutputCandidate*>* clusters = new std::vector<OutputCandidate*>();
		if (this->concurent == true){
			std::sort(clusterCandidates->begin(), clusterCandidates->end(),
					  [&](const OutputCandidate* lhs,
						  const OutputCandidate* rhs){
						  return lhs->score > rhs->score;
					  });
			for(unsigned int k = 0; k < clusterCandidates->size();){
				bool stillDisjoint = true;
				for(unsigned int j = 0; j < k; j++){
					stillDisjoint &= this->isDisjoint(clusterCandidates->at(k), clusterCandidates->at(j));
					if(stillDisjoint == false){						
						clusterCandidates->erase (clusterCandidates->begin()+k);
						break;
					}
				}
				if(stillDisjoint){
					k++;
				}
			}
			clusters = clusterCandidates;
		}else{
			for(unsigned int i = 0; i <clusterCandidates->size(); i++){
				if(clusterCandidates->at(i) != nullptr && (clusters->size() == 0 || clusters->at(0)->score < clusterCandidates->at(i)->score)){
					
					if(clusters->size() == 0){
						clusters->push_back(clusterCandidates->at(i));
					}else{
						clusters->at(0) = clusterCandidates->at(i);	
					}
				}
			}
		}
		
		// deleting the cluster from the dataset
		for(unsigned int k = 0; k < clusters->size(); k++){
			auto subspace = new std::vector<bool>;
			for(int i = 0; i < clusters->at(k)->item.size(); i++){
				subspace->push_back(clusters->at(k)->item[i]);
			}


			
			auto cube = HyperCube((this->medoids->at(clusters->at(k)->centroidNr)),this->width, subspace);

			std::vector<std::vector<float>*>* resC = new std::vector<std::vector<float>*>;

			for(int l = 0; l < this->data->size(); l++){
				auto point = this->data->at(l);
				if (cube.pointContained(point)){
					resC->push_back(point);
				}
			}
			auto result = std::make_pair(resC, subspace);
			int head = result.first->size()-1;
			for(int j = this->data->size()-1; j >=0  ;j-- ){
				if (this->data->at(j) == result.first->at(head)){
					auto temp = this->data->at(j);
					this->data->at(j) = this->data->at(this->data->size()-1);
					this->data->at(this->data->size()-1) = temp;
					this->data->pop_back();
					head--;
					if(head < 0){break;}
				}
			}

			res.push_back(result);
			if(this->data->size() <= 0){
				return res;
			}
		}
	}
	return res;
};



std::vector<boost::dynamic_bitset<>>* MineClus::findDimensions(std::vector<float>* centroid,
											std::vector<std::vector<float>* >* points, float width) {
	std::vector<boost::dynamic_bitset<>>* result = new std::vector<boost::dynamic_bitset<>>(points->size());

	for(int i = 0; i < points->size(); i++){
		auto point = boost::dynamic_bitset<>(centroid->size(),0);
		for(int j = 0; j < centroid->size(); j++){
			point[j] = abs(centroid->at(j)-points->at(i)->at(j)) < width;
		}
		result->at(i) = point;
	}

	return result;
}


std::vector<std::vector<float>*>* MineClus::pickRandom(int n) {
	std::vector<int> is = this->randInt(0,this->size()-1,n);
	std::vector<std::vector<float>*>* res = new std::vector<std::vector<float>*>;
	for(int i = 0; i < n; i++){
		res->push_back(this->data->at(is.at(i)));
	}
	return res;
}
