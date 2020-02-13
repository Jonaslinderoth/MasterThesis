#include <iostream>
#include <chrono>
#include <random>
#include <iostream>
#include "src/DOC_GPU/HyperCube.h"

int main ()
{
		auto a = new std::vector<std::vector<bool>*>;
		{auto aa = new std::vector<bool>{true, true};
		a->push_back(aa);};
	auto b = new std::vector<std::vector<float>*>;
    auto bb = new std::vector<float>{9,111};
	b->push_back(bb);
    auto centroid = new std::vector<float>{10, 10};
	auto centorids = new std::vector<std::vector<float>*>;
	centorids->push_back(centroid);
	auto c = pointsContained(a,b,centorids);
	for(int i = 0; i < c->size(); i++){
		std::cout << std::endl;
		for(int j = 0; j < c->at(i)->size(); j++){
			std:: cout << c->at(i)->at(j) << ", ";
		}
	}
}