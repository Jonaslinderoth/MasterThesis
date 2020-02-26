
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

	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
		{
			auto point = new std::vector<float>{1,1000};
			data->push_back(point);
		}
		{
			auto point = new std::vector<float>{0,100};
			data->push_back(point);
		}
		{
			auto point = new std::vector<float>{1,-100};
			data->push_back(point);
		}
		{
			auto point = new std::vector<float>{0,-1000};
			data->push_back(point);
		}
		DOCGPU d = DOCGPU(data);
		d.setSeed(2);
		std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = d.findCluster();
		std::cout << "is 4: " << res.first->size() << std::endl;

		std::cout << "is 2: " << res.second->size() << std::endl;;


}
