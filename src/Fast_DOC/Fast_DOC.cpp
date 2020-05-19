#include "Fast_DOC.h"
#include <stdexcept>
#include <math.h>
#include <algorithm>
#include "../DOC/HyperCube.h"


Fast_DOC::Fast_DOC(std::vector<std::vector<float>*>* input, float alpha, float beta, float width, unsigned int d0){
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



/**
 * Used for loading the data from the data reader into main memory.
 */
std::vector<std::vector<float>*>* Fast_DOC::initDataReader(DataReader* dr){
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



std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> Fast_DOC::findCluster(){
	float d = this->data->at(0)->size();
	float r = log2(2*d)/log2(1/(2*this->beta));
	float MAXITER = fmin(pow(d,2), pow(10,6));
	
	float m = fmin(MAXITER, pow((2/this->alpha),r) * log(4));

	unsigned int D_max_sum = 0;
	std::vector<bool>* D_max;
	std::vector<float>* p_max;

	
	unsigned int outer = (float)2/this->alpha;
	auto p_vec = this->pickRandom(outer);
	for(int i = 1; i < outer; i++){
		for(int j = 1; j < m; j++){
			std::vector<std::vector<float>*>* X = this->pickRandom(r);
			std::vector<bool>* D = this->findDimensions(p_vec->at(i),X,this->width);

			int Dsum = 0;
			std::for_each(D->begin(), D->end(), [&] (int n) {
					Dsum += n;
				});

			if(Dsum >= D_max_sum){
				D_max = D;
				D_max_sum = Dsum;
				p_max = p_vec->at(i);
			}
			if(D_max_sum >= d0){
				break;
			}			
		}
		if(D_max_sum >= d0){
			break;
		}
	}


	auto b_pD = HyperCube(p_max,this->width,D_max);
	std::vector<std::vector<float>*>* resC = new std::vector<std::vector<float>*>;
	
	for(int l = 0; l < this->data->size(); l++){
		auto point = this->data->at(l);
		if (b_pD.pointContained(point)){
			resC->push_back(point);
		}
	}
	auto result = std::make_pair(resC, D_max);
	return result;
};

std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> Fast_DOC::findKClusters(int k){
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



std::vector<bool>* Fast_DOC::findDimensions(std::vector<float>* centroid,
											std::vector<std::vector<float>* >* points, float width) {
	std::vector<bool>* result = new std::vector<bool>(centroid->size(), true);
	for(int i = 0; i < points->size(); i++){
		for(int j = 0; j < centroid->size(); j++){
			result->at(j) = result->at(j) && (abs(centroid->at(j)-points->at(i)->at(j)) < width);
		}
	}
	return result;
}


std::vector<std::vector<float>*>* Fast_DOC::pickRandom(int n) {
	std::vector<int> is = this->randInt(0,this->size()-1,n);
	std::vector<std::vector<float>*>* res = new std::vector<std::vector<float>*>;
	for(int i = 0; i < n; i++){
		res->push_back(this->data->at(is.at(i)));
	}
	return res;
}
