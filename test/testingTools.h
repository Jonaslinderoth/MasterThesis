#ifndef TESTINGTOOLS_H
#define TESTINGTOOLS_H
#include <math.h>
inline bool readBit(unsigned int value, unsigned int index){
	return ((value & (1 << index)) >> index);
}

inline double mu(unsigned int subSpaceSize, unsigned int support, float beta){
	return support*pow(((double) 1/beta),subSpaceSize);
}

#endif
