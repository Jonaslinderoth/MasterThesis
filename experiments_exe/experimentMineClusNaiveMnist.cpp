#include <iostream>
#include "../src/MineClusGPU/MineClusGPUnified.h"
#include "../experiments/genTestData.cpp"

int main(){
	auto data = getMnist();
	auto c = MineClusGPUnified(data);
	c.setDuplicatesVersion(Naive);
	c.findKClusters(10);
}

