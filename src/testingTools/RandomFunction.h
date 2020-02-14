/*
 * RandomFunction.h
 *
 *  Created on: Feb 13, 2020
 *      Author: mikkel
 */

#ifndef RANDOMFUNCTION_H_
#define RANDOMFUNCTION_H_

#include <random>

class RandomFunction {
private:
	static std::random_device rd;
	static std::mt19937 staticGen;
public:
	static void randomSeed();
	static void staticSetSeed(unsigned int seed);
	static float normalDistributionRandomFloat(float mean , float variance);
	static unsigned int randomInteger();
	static float uniformRandomFloat(float lowest , float max);

};


#endif /* RANDOMFUNCTION_H_ */
