#include "src/randomCudaScripts/arrayEqual.h"
#include "src/DOC_GPU/DOCGPU_Kernels.h"
#include "src/DOC_GPU/pointsContainedDevice.h"

#include <iostream>


bool areTheyEqual_d(bool* a_d ,bool* b_d , unsigned long lenght){
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

	//std::cout << "count " << count << " countBad " << countBad << std::endl;
	//std::cout << "countZero " << countZero << " countOne " << countOne << std::endl;

	delete(a_h);
	delete(b_h);
	return ret;
}


bool areTheyEqual_d(unsigned int* a_d ,unsigned int* b_d , unsigned long lenght){
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
	//std::cout << std::endl;

	//std::cout << "count " << count << " countBad " << countBad << std::endl;

	delete(a_h);
	delete(b_h);
	return ret;
}


bool areTheyEqual_h(std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> a_h ,
				    std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> b_h,
				    bool print){
	unsigned long count = 0;
	unsigned long countEqual = 0;

	bool ret = true;

	std::vector<std::vector<bool>*>* a_first = a_h.first;
	std::vector<std::vector<bool>*>* b_first =b_h.first;

	if(a_first == nullptr or b_first == nullptr){
		if(a_first != b_first){
			if(print){
				std::cout << "01 nullptr" << std::endl;
			}
			return false;
		}
	}else{
		if(a_first->size() != b_first->size()){
			if(print){
				std::cout << "02 size not maching" << std::endl;
			}
			ret = false;
		}
		for(int i = 0 ; i < a_first->size() ; ++i){

			if(a_first->at(i) == nullptr or b_first->at(i) == nullptr){
				if(a_first->at(i) != b_first->at(i)){
					if(print){
						std::cout << "03 size not maching" << std::endl;
					}
					ret=  false;
				}
			}else{
				if(a_first->at(i)->size() != b_first->at(i)->size()){
					if(print){
						std::cout << "04: inner size not matching in: " << i << std::endl;
					}
					ret = false;
				}
				for(int j = 0 ; j < a_first->at(i)->size() ; ++j){
					count++;
					if(a_first->at(i)->at(j) != b_first->at(i)->at(j)){

						if(print or true){
							std::cout << "05: i " << i << " j " << j << " " << a_first->at(i)->at(j) << " != " << b_first->at(i)->at(j) << std::endl;
						}
						ret = false;
					}else
					{
						countEqual++;
					}
				}

			}
		}
	}
	if(print){
		std::cout << "count " << count << " countEqual " << countEqual << std::endl;
	}
	count = 0;
	countEqual = 0;
	std::vector<unsigned int>* a_second = a_h.second;
    std::vector<unsigned int>* b_second = b_h.second;

    if(a_second == nullptr or b_second == nullptr){
    	if(a_second != b_second){
    		if(print){
				std::cout << "10 nullptr" << std::endl;
			}
    		return false;
    	}
    }else{
    	if(a_second->size() != b_second->size()){
    		if(print){
				std::cout << "11 size not maching" << std::endl;
			}
    		ret =  false;
		}

		for(int i = 0 ; i < a_second->size() ; ++i){
			count++;
			if(a_second->at(i) != b_second->at(i)){

				if(print){
					std::cout << "12: i: "<< i << " " << a_second->at(i)  << " != " << b_second->at(i) << std::endl;
				}
				ret = false;
			}else{
				countEqual++;
			}
		}
    }

    if(print){
		std::cout << "count " << count << " countEqual " << countEqual << std::endl;
	}
	return ret;
}

bool printArray(unsigned int* a_d , unsigned long lenght , unsigned long maxHowMuch){

	unsigned int * a_h;

	a_h = (unsigned int*)malloc(lenght*sizeof(unsigned int));

	cudaMemcpy(a_h, a_d, lenght*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	unsigned long min = lenght;

	if(lenght > maxHowMuch ){
		min = maxHowMuch;
	}

	for(int i = 0 ; i < min ; i++){
		std::cout << a_h[i] << " ";
	}
	std::cout << std::endl;

	return true;
}

bool printArray(bool* a_d , unsigned long lenght , unsigned long maxHowMuch){

	bool * a_h;

	a_h = (bool*)malloc(lenght*sizeof(bool));

	cudaMemcpy(a_h, a_d, lenght*sizeof(bool), cudaMemcpyDeviceToHost);

	unsigned long min = lenght;

	if(lenght > maxHowMuch ){
		min = maxHowMuch;
	}

	for(int i = 0 ; i < min ; i++){
		if(a_h[i]){
			std::cout << "true ";
		}else{
			std::cout << "false ";
		}

	}
	std::cout << std::endl;

	return true;
}

bool printPair_h(std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> a_h , const unsigned long maxLenght){

	std::vector<std::vector<bool>*>* a_first = a_h.first;

	std::cout << "first" << std::endl;
	unsigned long lenght = 0;
	for(int i = 0 ; i < a_first->size() ; ++i){
		for(int j = 0 ; j < a_first->at(i)->size() ; ++j){
			if(maxLenght < lenght){
				break;
			}
			lenght++;

			std::cout << a_first->at(i)->at(i) << " ";
		}
		if(maxLenght < lenght){
			break;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "second" << std::endl;
	std::vector<unsigned int>* a_second = a_h.second;
	for(unsigned long i = 0 ; i < a_second->size() ; ++i){
		if(maxLenght < lenght){
			break;
		}
		lenght++;
		std::cout << a_second->at(i) << " ";
	}
	std::cout << std::endl;
	return true;

}




std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> pointsContained(std::vector<std::vector<bool>*>* dims,
																					   std::vector<std::vector<float>*>* data,
																					   std::vector<unsigned int>* centroids,
																					   int m,
																					   float width,
																					   unsigned long version,
																					   unsigned long garbage,
																					   unsigned long breakIntervall){


    int block_size = 1024;

    // Calculaating sizes
	std::size_t point_dim = data->at(0)->size();
	std::size_t no_of_points = data->size();
	std::size_t no_of_dims = dims->size();
	std::size_t no_of_centroids = centroids->size();

	std::size_t floats_in_data = point_dim * no_of_points;
	std::size_t bools_in_dims = no_of_dims * point_dim;
	std::size_t bools_in_output = no_of_points * no_of_dims;
	std::size_t ints_in_output_count = no_of_dims;

	std::size_t size_of_data = floats_in_data*sizeof(float);
	std::size_t size_of_dims = bools_in_dims*sizeof(bool);
	std::size_t size_of_centroids = no_of_centroids*sizeof(unsigned int);
	std::size_t size_of_output = bools_in_output*sizeof(bool);
	std::size_t size_of_output_count = ints_in_output_count*sizeof(unsigned int);

	// allocating on the host
	float* data_h = (float*) malloc(size_of_data);
	bool* dims_h = (bool*) malloc(size_of_dims);
	unsigned int* centroids_h = (unsigned int*) malloc(size_of_centroids);
	bool* output_h = (bool*) malloc(size_of_output);
	unsigned int* output_count_h = (unsigned int*) malloc(size_of_output_count);

	// filling data array
	for(int i= 0; i < no_of_points; i++){
		for(int j = 0; j < point_dim; j++){
			data_h[i*point_dim+j] = data->at(i)->at(j);
		}
	}

	// filling dims array
	for(int i= 0; i < no_of_dims; i++){
		for(int j = 0; j < point_dim; j++){
			dims_h[i*point_dim+j] = dims->at(i)->at(j);
		}
	}

	// filling centroid array
	for(int i= 0; i < no_of_centroids; i++){
		centroids_h[i] = centroids->at(i);
	}

	// allocating on device
	float* data_d;
	bool* dims_d;
	unsigned int* centroids_d;
	bool* output_d;
	unsigned int* output_count_d;

	bool* garbageCleaner_h = (bool*) malloc(size_of_output);
	unsigned int* garbageCleanerCount_h = (unsigned int*) malloc(size_of_output_count);

	for(unsigned int i = 0 ; i < bools_in_output ; i++){
		garbageCleaner_h[i] = (bool)garbage;

	}
	for(unsigned int i = 0 ; i < ints_in_output_count ; i++){
		garbageCleanerCount_h[i] = (unsigned int)garbage;

	}




	cudaMalloc((void **) &data_d, size_of_data);
	cudaMalloc((void **) &dims_d, size_of_dims);
	cudaMalloc((void **) &centroids_d, size_of_centroids);
	cudaMalloc((void **) &output_d, size_of_output);
	cudaMalloc((void **) &output_count_d, size_of_output_count);


	cudaMemcpy(output_d, garbageCleaner_h, size_of_output, cudaMemcpyHostToDevice);
	cudaMemcpy(output_count_d, garbageCleanerCount_h, size_of_output_count, cudaMemcpyHostToDevice);
	//Copy from host to device

	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);
	cudaMemcpy(dims_d, dims_h, size_of_dims, cudaMemcpyHostToDevice);
	cudaMemcpy(centroids_d, centroids_h, size_of_centroids, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    (cudaStreamCreate(&stream));

	if(version == 0){
		// Call kernel

		pointsContainedKernelNaive(ceil((no_of_dims)/(float)block_size),
								   block_size,
								   stream,
								   data_d,
								   centroids_d,
								   dims_d,
								   output_d,
								   output_count_d,
								   width,
								   point_dim,
								   no_of_points,
								   no_of_dims,
								   m);


	}else if(version == 1){
		pointsContainedKernelSharedMemory(ceil((no_of_dims)/(float)block_size),
		                        block_size,
								  stream,
								  data_d,
								  centroids_d,
								  dims_d,
								  output_d,
								  output_count_d,
								  width,
								  point_dim,
								  no_of_points,
								  no_of_dims,
								  m,
								  no_of_centroids);
	}else if(version == 2){
		pointsContainedKernelSharedMemoryFewBank(ceil((no_of_dims)/(float)block_size),
                                            block_size,
										  stream,
										  data_d,
										  centroids_d,
										  dims_d,
										  output_d,
										  output_count_d,
										  width,
										  point_dim,
										  no_of_points,
										  no_of_dims,
										  m,
										  no_of_centroids);
	}else if(version == 3){
		pointsContainedKernelSharedMemoryFewerBank(ceil((no_of_dims)/(float)block_size),
                                                block_size,
												 stream,
												 data_d,
												 centroids_d,
												 dims_d,
												 output_d,
												 output_count_d,
												 width,
												 point_dim,
												 no_of_points,
												 no_of_dims,
												 m,
												 no_of_centroids);
	}else if(version == 4){
		pointsContainedKernelFewPoints(ceil((no_of_dims)/(float)block_size),
                                       block_size,
                                       stream,
                                       data_d,
                                       centroids_d,
                                       dims_d,
                                       output_d,
                                       output_count_d,
                                       width,
                                       point_dim,
                                       no_of_points,
                                       m,
                                       no_of_centroids);
	}else if(version == 5){
		// Call kernel
		pointsContainedKernelNaiveBreak(ceil((no_of_dims)/(float)block_size),
								   	   block_size,
								   	   stream,
								   	   data_d,
								   	   centroids_d,
								   	   dims_d,
								   	   output_d,
								   	   output_count_d,
								   	   width,
								   	   point_dim,
								   	   no_of_points,
								   	   no_of_dims,
								   	   m,
								   	   breakIntervall);

	}

    (cudaStreamDestroy(stream));

	// copy from device
	cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost);
	cudaMemcpy(output_count_h, output_count_d, size_of_output_count, cudaMemcpyDeviceToHost);

	// construnct output
	auto output =  new std::vector<std::vector<bool>*>;
	auto output_count =  new std::vector<unsigned int>;


	for(int i = 0; i < no_of_dims; i++){
		auto a =  new std::vector<bool>;
		for(int j = 0; j < no_of_points; j++){
			a->push_back(output_h[i*no_of_points+j]);
		}
		output->push_back(a);
	}

	for(int i = 0; i < no_of_dims; i++){
		output_count->push_back(output_count_h[i]);
	}


	cudaFree(data_d);
	cudaFree(dims_d);
	cudaFree(centroids_d);
	cudaFree(output_d);
	cudaFree(output_count_d);
	free(data_h);
	free(dims_h);
	free(centroids_h);
	free(output_h);
	free(output_count_h);
	free(garbageCleaner_h);
	free(garbageCleanerCount_h);


	return std::make_pair(output,output_count);
};

