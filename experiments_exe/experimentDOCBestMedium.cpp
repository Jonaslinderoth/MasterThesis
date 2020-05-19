#include <iostream>
#include "../src/DOC_GPU/DOCGPU.h"
#include "../experiments/genTestData.cpp"
#include "../src/randomCudaScripts/dummyKernel.h"

int main(){
	dummyKernelWrapper();
	genTestData();
	DataReader* dr = new DataReader("testData/mediumDataSet");
	auto c = DOCGPU(dr);
	c.setNumberOfSamples(4096*2);
	c.setWidth(15);
	c.setSeed(1);
	c.setFindDimVersion(chunksFindDim);
	auto res = c.findKClusters(10);
	// std::cout << res.size() << std::endl;
	// for(unsigned int i = 0; i < res.size(); i++){
	// 	std::cout << res.at(i).first->size() << std::endl;
	// }
}

