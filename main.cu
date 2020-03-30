
#include <iostream>
#include <random>
#include "src/MineClusGPU/MineClusGPU.h"
#include <vector>




int main ()
{
	auto data = new std::vector<std::vector<float>*>;
	auto point1 = new std::vector<float>({1,2});
	data->push_back(point1);

	for(int i = 0; i < 10; i++){
		auto point = new std::vector<float>({1000,20000});
		data->push_back(point);
	}
	
	for(int i = 0; i < 10; i++){
		auto point = new std::vector<float>({1,20000});
		data->push_back(point);
	}
	
	auto c = MineClusGPU(data);


	auto res = c.findKClusters(1);
	return 0;
}
