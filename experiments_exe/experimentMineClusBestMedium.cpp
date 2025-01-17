#include <iostream>
#include "../src/MineClusGPU/MineClusGPU.h"
#include "../experiments/genTestData.cpp"
#include "../src/randomCudaScripts/dummyKernel.h"

int main(){
	dummyKernelWrapper();
	genTestData();
	DataReader* dr = new DataReader("testData/mediumDataSet");
	auto c = MineClusGPU(dr);
	c.setDuplicatesVersion(Hash);
	c.setCountSupportVersion(SmemCount);
	c.setWidth(15);
	c.setSeed(11);
	auto res = c.findKClusters(10);
	// std::cout << res.size() << std::endl;
	// for(unsigned int i = 0; i < res.size(); i++){
	// 	std::cout << res.at(i).first->size() << std::endl;
	// 	for(unsigned int j = 0; j < res.at(i).second->size(); j++){
	// 		std::cout << res.at(i).second->at(j) << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
}


