#include <random>
#include <vector>
#include <iostream>
#include "src/Fast_DOCGPU/Fast_DOCGPUnified.h"
#include "src/DOC_GPU/DOCGPUnified.h"
#include "src/testingTools/DataGeneratorBuilder.h"

int main(){
	DataGeneratorBuilder dgb;

	std::string fineName = "test11";

	bool res = dgb.buildUClusters(fineName,10,3,15,2,1,0);
	

	DataReader* dr = new DataReader(fineName);

	while(dr->isThereANextPoint()){
		std::vector<float>* point = dr->nextPoint();
		for(std::vector<float>::iterator iter = point->begin() ; iter != point->end() ; ++iter){
			std::cout << std::to_string(*iter) << " ";
		}
		std::cout << std::endl;
	}


	return res;
}
