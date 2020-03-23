#ifndef TESTINGTOOLS_H
#define TESTINGTOOLS_H

inline bool readBit(unsigned int value, unsigned int index){
	return ((value & (1 << index)) >> index);
}

#endif
