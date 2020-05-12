#ifndef TESTINGTOOLS_H
#define TESTINGTOOLS_H
#include <math.h>
inline bool readBit(unsigned int value, unsigned int index){
	return ((value & (1 << index)) >> index);
}

inline double mu(unsigned int subSpaceSize, unsigned int support, float beta){
	return support*pow(((double) 1/beta),subSpaceSize);
}


inline bool pointEQ(std::vector<float>* a1, std::vector<float>* a2){
	bool output = true;
	//EXPECT_EQ(a1->size(), a2->size());
	if(a1->size() != a2->size()){
		return false;
	}

	for(int j = 0; j < a2->size(); j++){
		auto b1 = a1->at(j);
		auto b2 = a2->at(j);
		output &= abs(b1 - b2) <= 0.0001;
		if(!output){
			break;
		}
	}
	//EXPECT_TRUE(output);
	
	return output;
}

inline bool disjoint(std::vector<std::vector<float>*>* a1, std::vector<std::vector<float>*>* a2){
	bool output = true;

	for(int i = 0; i < a1->size(); i++){
		for(int j = 0; j < a2->size(); j++){
			std::vector<float>* b1 = a1->at(i);
			std::vector<float>* b2 = a2->at(j);
			bool eq = pointEQ(b1, b2);
			output &= !eq;		
		}
	}
	return output;
}


inline bool equal(std::vector<std::vector<float>*>* a1, std::vector<std::vector<float>*>* a2){
	bool output = true;
	EXPECT_EQ(a1->size(), a2->size());
	
	for(int i = 0; i < a1->size(); i++){
		output = false;
		for(int j = 0; j < a2->size(); j++){
			auto b1 = a1->at(i);
			auto b2 = a2->at(j);
			auto eq = pointEQ(b1, b2);
			if(eq){output = eq; break;}
		}			
	}
	return output;
}

#endif
