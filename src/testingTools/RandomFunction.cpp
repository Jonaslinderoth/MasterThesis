/*
 * RandomFunction.cpp
 *
 *  Created on: Feb 13, 2020
 *      Author: mikkel
 */

#include <src/testingTools/RandomFunction.h>
#include <iostream>

std::random_device   RandomFunction::rd;
std::mt19937 RandomFunction::staticGen(RandomFunction::rd());


void RandomFunction::randomSeed(){
	//std::cout << "random seed" << std::endl;
	std::random_device rd;
	staticGen = std::mt19937(rd());
}

void RandomFunction::staticSetSeed(unsigned int seed){
	//std::cout << "seed set" << std::endl;
	staticGen.seed(seed);
}

float RandomFunction::normalDistributionRandomFloat(float mean , float variance){
	std::normal_distribution<float> distribution(mean,sqrt(variance));
	float res = distribution(staticGen);
	//std::cout << "mean " << mean << " var " << variance << std::endl;
	return res;
}

float RandomFunction::uniformRandomFloat(float lowest , float max){
	std::uniform_real_distribution<> distribution(lowest,max);
	float res = distribution(staticGen);
	//std::cout << "uniformRandomFloat " << res << std::endl;
	return res;
}
unsigned int RandomFunction::randomInteger(){
	std::uniform_int_distribution<unsigned int> distibuition(0,2147483647);
	unsigned int res = distibuition(staticGen);
	//std::cout << "randomInteger " << res << std::endl;
	return res;
}
