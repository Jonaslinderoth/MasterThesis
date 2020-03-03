#include "src/randomCudaScripts/arrayEqual.h"
#include <iostream>


bool areTheyEqual(bool* a_d ,bool* b_d , unsigned long lenght){
	bool * a_h;
	bool * b_h;

	a_h = (bool*)malloc(lenght*sizeof(bool));
	b_h = (bool*)malloc(lenght*sizeof(bool));


	cudaMemcpy(a_h, a_d, lenght*sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(b_h, b_d, lenght*sizeof(bool), cudaMemcpyDeviceToHost);
	bool ret = true;
	unsigned long count = 0;
	unsigned long countBad = 0;
	unsigned long countZero = 0;
	unsigned long countOne = 0;

	for(unsigned long i = 0 ; i < lenght ; i++){
		if(a_h[i] != b_h[i]){
			ret = false;
			countBad++;
			//std::cout << "*" ;
			//std::cout << "at: " << i << " " << a_h[i] << " != " << b_h[i] << std::endl;
		}else{
			//std::cout << "=";
		}
		count++;
		if(a_h[i] == 0)
		{
			countZero++;
		}else{
			countOne++;
		}

	}
	//std::cout << std::endl;

	std::cout << "count " << count << " countBad " << countBad << std::endl;
	std::cout << "countZero " << countZero << " countOne " << countOne << std::endl;

	delete(a_h);
	delete(b_h);
	return ret;
}


bool areTheyEqual(unsigned int* a_d ,unsigned int* b_d , unsigned long lenght){
	unsigned int * a_h;
	unsigned int * b_h;

	a_h = (unsigned int*)malloc(lenght*sizeof(unsigned int));
	b_h = (unsigned int*)malloc(lenght*sizeof(unsigned int));


	cudaMemcpy(a_h, a_d, lenght*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(b_h, b_d, lenght*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	bool ret = true;
	unsigned long count = 0;
	unsigned long countBad = 0;

	for(unsigned long i = 0 ; i < lenght ; i++){
		if(a_h[i] != b_h[i]){
			ret = false;
			countBad++;
			//std::cout << "*" ;
			//std::cout << "at: " << i << " " << a_h[i] << " != " << b_h[i] << std::endl;
		}else{
			//std::cout << "=";
		}
		count++;
	}
	//			std::cout << std::endl;

	std::cout << "count " << count << " countBad " << countBad << std::endl;

	delete(a_h);
	delete(b_h);
	return ret;
}
