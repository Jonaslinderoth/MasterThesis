#include <iostream>
#include "../src/MineClusGPU/MineClusGPU.h"
#include "../experiments/genTestData.cpp"

int main(){
	genTestData();
	DataReader* dr = new DataReader("testData/LargeDataSet");
	auto c = MineClusGPU(dr);
	c.setDuplicatesVersion(Naive);
	c.findKClusters(10);
}
