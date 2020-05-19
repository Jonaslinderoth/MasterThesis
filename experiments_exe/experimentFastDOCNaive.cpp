#include <iostream>
#include "../src/Fast_DOCGPU/Fast_DOCGPU.h"
#include "../experiments/genTestData.cpp"

int main(){
	genTestData();
	DataReader* dr = new DataReader("testData/smallDataSet");
	auto c = Fast_DOCGPU(dr);
	c.setWidth(15);
	c.setSeed(1);
	auto res = c.findKClusters(10);
	
	// std::cout << res.size() << std::endl;
	// for(unsigned int i = 0; i < res.size(); i++){
	// 	std::cout << res.at(i).first->size() << std::endl;
	// }
}
