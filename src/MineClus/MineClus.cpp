#include "MineClus.h"
#include <boost/dynamic_bitset.hpp>
#include <stdexcept>
#include <math.h>
#include <algorithm>
#include "../DOC/HyperCube.h"
#include "../MuApriori/MuApriori.h"



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


std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> MineClus::findCluster(){

	
	float d = this->data->at(0)->size();
	unsigned int min_supp = this->alpha*this->data->size();
	unsigned int outer = (float)2/this->alpha;
	std::vector<std::vector<float>*>* medioids = this->pickRandom(outer);
	MuApriori* muApriori;
	for(int i = 0; i < outer; i++){
		// compute itemSet
		std::vector<boost::dynamic_bitset<>>* itemSet = this->findDimensions(medioids->at(i),
							  this->data, this->width);
		// call apriori
		if(i == 0){
			muApriori = new MuApriori(itemSet, min_supp, this->beta);
		}else{
			muApriori->setItemSet(itemSet);
		}
		muApriori->findBest(1);
	}
	auto best = muApriori->getBest();

	delete muApriori;

	auto subspace = new std::vector<bool>;
	
	for(int i = 0; i < best->at(0)->item.size(); i++){
		subspace->push_back(best->at(0)->item[i]);
	}

	
	auto cube = HyperCube((medioids->at(best->at(0)->centroidNr)),this->width, subspace);
	std::vector<std::vector<float>*>* resC = new std::vector<std::vector<float>*>;

	for(int l = 0; l < this->data->size(); l++){
		auto point = this->data->at(l);
		if (cube.pointContained(point)){
			resC->push_back(point);
		}
	}
	
	auto result = std::make_pair(resC, subspace);
	return result;

};

std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> MineClus::findKClusters(int k){
	auto res = std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>>();
	for(int i = 0; i < k; i++){
		if(this->data->size() <= 0) {break;}
		auto cluster = this->findCluster();
		int head = cluster.first->size()-1;
		for(int j = this->data->size()-1; j >=0  ;j-- ){
			if (this->data->at(j) == cluster.first->at(head)){
				auto temp = this->data->at(j);
				this->data->at(j) = this->data->at(this->data->size()-1);
				this->data->at(this->data->size()-1) = temp;
				this->data->pop_back();
				head--;
				if(head < 0){break;}
			}
		}
		res.push_back(cluster);
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
