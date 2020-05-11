
#include "src/Fast_DOCGPU/Fast_DOCGPUnified.h"
#include <random>

int main(){
	unsigned int dim = 10;
	unsigned int numPoints = 1000;
	auto data =  new std::vector<std::vector<float>*>();

	std::default_random_engine generator;
	generator.seed(1);
	std::normal_distribution<float> cluster(5.0,2.0);
	std::uniform_int_distribution<> outlier(-10000,10000);
	
	
	for(int i = 0; i < numPoints; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			if(j%2 == 0){
				point->push_back(cluster(generator));
			}else{
				point->push_back(outlier(generator));
			}
		}
		data->push_back(point);
	}

	auto c = new Fast_DOCGPUnified(data);
	c->setSeed(2);

	
	auto res = c->findCluster();
	

	for(unsigned int i = 0; i < data->size(); i++){
		delete data->at(i);
	}
	delete data;
	res.first->clear();
	delete res.first;
	delete res.second;
	delete c;
}
