
#include "DOC.h"
#include "HyperCube.h"
#include <iostream>
#include <math.h>       /* log */
#include <algorithm>

DOC::DOC(): DOC(new std::vector<std::vector<float>*>){}

DOC::DOC(std::vector<std::vector<float>*>* input): DOC(input, 0.1, 0.25, 15) {}

DOC::DOC(std::vector<std::vector<float>*>* input, float alpha, float beta, float width){
	this->data = input;
	this->alpha = alpha;
	this->width = width;
	this->beta = beta;

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
			int Dsum = 0;
			std::for_each(D->begin(), D->end(), [&] (int n) {
			    Dsum += D->at(n);
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
			// todo: delete X, delete D and C if not the current best,

		}

	}

	auto result = std::make_pair(resC, resD);
	return result;
}

std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> findKClusters(int k){
	auto res = std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>>();
	for(int i = 0; i < k; i++){
		//res.push_back(this->findCluster());
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
			result->at(j) = result->at(j) && (centroid->at(j)-points->at(i)->at(j) < width);
		}
	}
	return result;
}


