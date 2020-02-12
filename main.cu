#include <iostream>
#include <chrono>
#include <random>
#include <iostream>
#include "src/DOC_GPU/HyperCube.h"

int main ()
{
	auto ps = new std::vector<std::vector<float>*>;
	{auto p = new std::vector<float>{1,1,1,1};
		ps->push_back(p);}
	{auto p = new std::vector<float>{1000,1000,1000,1000};
	ps->push_back(p);}
	auto xss = std::vector<std::vector<std::vector<float>*>*>();
	auto xs = new std::vector<std::vector<float>*>;
	auto x1 = new std::vector<float>{1,2,1000,1};
	auto x2 = new std::vector<float>{2,1, 1000,1};
	xs->push_back(x1);
	xs->push_back(x2);
	xss.push_back(xs);

	
	auto res = findDimmensions(ps, xss);

		std::cout << std::endl;
	for(int i = 0; i < res->size(); i++){
		for(int j = 0; j < res->at(i)->size(); j++){
			std::cout << res->at(i)->at(j) << ", ";
		}
		std::cout << std::endl;
	}
}