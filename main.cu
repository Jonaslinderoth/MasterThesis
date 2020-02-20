
#include <iostream>
#include <chrono>
#include <random>
#include <iostream>
#include "src/DOC_GPU/DOCGPU.h"
#include "src/Clustering.h"
#include "test/testData.h"
#include <vector>
int main ()
{
	
	{std::vector<std::vector<float>*>* data = data_4dim2cluster();

	Clustering* c = new DOCGPU(data);
	c->setWidth(6);
	c->setSeed(100);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = c->findCluster();
	
	std::cout << res.first->size() << ", " << res.second->size()  << std::endl;}

		{std::vector<std::vector<float>*>* data = data_4dim2cluster();

	Clustering* c = new DOCGPU(data);
	c->setWidth(6);
	c->setSeed(100);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = c->findCluster();
	
	std::cout << res.first->size() << ", " << res.second->size()  << std::endl;}
}