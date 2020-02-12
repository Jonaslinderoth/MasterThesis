
#include "DOC.h"
#include "HyperCube.h"
#include <iostream>
#include <math.h>       /* log */
#include <algorithm>

DOC::DOC(): DOC(new std::vector<std::vector<float>*>){}

DOC::DOC(std::vector<std::vector<float>*>* input): DOC(input, 0.1, 0.25, 15) {}

DOC::DOC(DataReader* dr): DOC(initDataReader(dr)){};

std::vector<std::vector<float>*>* DOC::initDataReader(DataReader* dr){
		auto size = dr->getSize();
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>(0);
	data->reserve(size);
	while(dr->isThereANextBlock()){
		std::vector<std::vector<float>*>* block = dr->next();
		data->insert(data->end(), block->begin(), block->end());
		delete block;
	}
	return data;
};

DOC::DOC(std::vector<std::vector<float>*>* input, float alpha, float beta, float width){
	this->data = input;
	this->alpha = alpha;
	this->width = width;
	this->beta = beta;
	this->gen.seed(this->seed);

}

std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> DOC::findCluster() {
	float d = this->data->at(0)->size();
	float r = log2(2*d)/log2(1/(2*this->beta));
	float m = pow((2/this->alpha),2) * log(4);

	auto resD = new std::vector<bool>;
	std::vector<std::vector<float>*>* resC = new std::vector<std::vector<float>*>;
	float maxValue = 0;
	for(int i = 1; i < (float)2/this->alpha; i++){
		auto p_vec = this->pickRandom(1);
		auto p = p_vec->at(0);
		delete p_vec;
		for(int j = 1; j < m; j++){
			std::vector<std::vector<float>*>* X = this->pickRandom(r);
			auto D = this->findDimensions(p,X,this->width);
			std::vector<std::vector<float>*>* C = new std::vector<std::vector<float>*>;
			auto b_pD = HyperCube(p,this->width,D);

			for(int l = 0; l < this->data->size(); l++){
				auto point = this->data->at(l);
				if (b_pD.pointContained(point)){
					C->push_back(point);
				}

			}

			// to not find too small clusters, then we prefer it to pick
			// larger clusters with larger dimensions
			if(C->size() < this->alpha*this->data->size()){ 
				delete C;
				delete D;
				D = new std::vector<bool>(d,false);
				C = new std::vector<std::vector<float>*>;
			}

			int Dsum = 0;

			std::for_each(D->begin(), D->end(), [&] (int n) {
					Dsum += n;
			});

			
			float current = mu(C->size(), Dsum);
			if(current > maxValue){
				resD = D;
				resC = C;
				maxValue = current;
			}else{
				delete C;
				delete D;
			}

			delete X;
		}
	  
	}
	auto result = std::make_pair(resC, resD);
	return result;
}

std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > DOC::findKClusters(int k){
	auto res = std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> >();
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
}

bool DOC::addPoint(std::vector<float>* point) {
	data->push_back(point);
	return true;
}

int DOC::size() {
	return data->size();
}


std::vector<std::vector<float>*>* DOC::pickRandom(int n) {
	std::vector<int> is = this->randInt(0,this->size()-1,n);
	std::vector<std::vector<float>*>* res = new std::vector<std::vector<float>*>;
	for(int i = 0; i < n; i++){
		res->push_back(this->data->at(is.at(i)));
	}
	return res;
}

std::vector<bool>* DOC::findDimensions(std::vector<float>* centroid,
		std::vector<std::vector<float>* >* points, float width) {
	std::vector<bool>* result = new std::vector<bool>(centroid->size(), true);
	for(int i = 0; i < points->size(); i++){
		for(int j = 0; j < centroid->size(); j++){
			result->at(j) = result->at(j) && (abs(centroid->at(j)-points->at(i)->at(j)) < width);
		}
	}
	return result;
}


