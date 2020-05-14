#include "ExperimentRemoveDuplicates.h"
#include <random>
#include "../src/MineClusGPU/FindDublicates.h"
#include "../src/randomCudaScripts/Utils.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>

void ExperimentRemoveDuplicates::start(){
	unsigned int c = 0;
	unsigned int numberOfCandidates = 1000000;
	unsigned int dim =                1000;

	// count tests
	for(unsigned int j = 100; j < numberOfCandidates; j *= 2){
		for(unsigned int k = 32; k < dim; k*= 2){
			c++;
		}	
	}		

	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	Experiment::addTests(c);
	Experiment::start();

	
	// Do experiments
	for(unsigned int i = 100; i < numberOfCandidates; i *= 2){
		for(unsigned int k = 32; k < dim; k*= 2){
			// Calculate sizes
			size_t sizeOfCandidates = i*(k/32)*sizeof(unsigned int);
			size_t sizeOfOutput = i*sizeof(bool);
			size_t sizeOfAlreadyDeleted = i*sizeof(bool);


			unsigned int* candidates;
			bool* alreadyDeleted;
			bool* output_1;
			bool* output_2;
			bool* output_3;
			bool* output_4;
			// Allocate unified memory
			checkCudaErrors(cudaMallocManaged((void**) &candidates, sizeOfCandidates));
			checkCudaErrors(cudaMallocManaged((void**) &alreadyDeleted, sizeOfAlreadyDeleted));
			checkCudaErrors(cudaMallocManaged((void**) &output_1, sizeOfOutput));
			checkCudaErrors(cudaMallocManaged((void**) &output_2, sizeOfOutput));
			checkCudaErrors(cudaMallocManaged((void**) &output_3, sizeOfOutput));
			checkCudaErrors(cudaMallocManaged((void**) &output_4, sizeOfOutput));
																
				
			   
			// Generate candidates
			std::vector<std::vector<unsigned int>*> duplicates = std::vector<std::vector<unsigned int>*>();
			for(unsigned int v = 0; v < 100; v++){
				candidates[v] = v+1;
				for(unsigned int kk = 1; kk < k/32; kk++){
					unsigned int block = std::uniform_int_distribution<unsigned int>{}(rng);
					candidates[v+kk*i] = block;			
				}
				duplicates.push_back(new std::vector<unsigned int>{v});
			}

			for(unsigned int a = 1; a < 1/100; a++){
				for(unsigned int b = 0; b < 100; b++){
					duplicates.at(b)->push_back(a*100+b);
					for(unsigned int c = 0; c < k/32; c++){
						candidates[(a*100+b) + c*i] = candidates[(b) + c*i];
					}
				}
			}

			// for(unsigned int n = 0; n < duplicates.size(); n++){
			// 	for(unsigned int m = 0; m < duplicates.at(n)->size(); m++){
			// 		std::cout << duplicates.at(n)->at(m) << " ";
			// 	}
			// 	std::cout << std::endl;
			// }			
			// std::cout << std::endl;
			
			// Generate to be deleted
			for(unsigned int ii = 0; ii < i; ii++){
				alreadyDeleted[ii] = 0;
				// maybe change to a procentage of the points
			}

			cudaStream_t stream;
			checkCudaErrors(cudaStreamCreate(&stream));
			cudaEvent_t start_e, stop_e;
			cudaEventCreate(&start_e);
			cudaEventCreate(&stop_e);
			float millis_1 = 0;
			float millis_2 = 0;
			float millis_3 = 0;
			float millis_4 = 0;

			// Naive
			{
				// Prefetch
				checkCudaErrors(cudaMemPrefetchAsync(candidates, sizeOfCandidates, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(output_1, sizeOfOutput, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(alreadyDeleted, sizeOfAlreadyDeleted, 0, stream));
					
				// Start timer
				checkCudaErrors(cudaEventRecord(start_e, stream));
				// Call kernel
				checkCudaErrors(cudaMemsetAsync(output_1, 0, sizeOfOutput, stream));
				findDublicatesWrapper(ceilf((float)numberOfCandidates/1024),
									  1024,
									  stream,
									  candidates,
									  i, // number of candidates
									  k, // dim
									  alreadyDeleted,
									  output_1,
									  Naive);
						 
				// End timer
				checkCudaErrors(cudaEventRecord(stop_e, stream));
				checkCudaErrors(cudaEventSynchronize(stop_e));
				cudaEventElapsedTime(&millis_1, start_e, stop_e);
			}


			// Breaking
			{
				// Prefetch
				checkCudaErrors(cudaMemPrefetchAsync(candidates, sizeOfCandidates, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(alreadyDeleted, sizeOfAlreadyDeleted, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(output_2, sizeOfOutput, 0, stream));
					
				// Start timer
				checkCudaErrors(cudaEventRecord(start_e, stream));
				// Call kernel
				checkCudaErrors(cudaMemsetAsync(output_2, 0, sizeOfOutput, stream));
				findDublicatesWrapper(ceilf((float)numberOfCandidates/1024),
									  1024,
									  stream,
									  candidates,
									  i, // number of candidates
									  k, // dim
									  alreadyDeleted,
									  output_2,
									  Breaking);
						 
				// End timer
				checkCudaErrors(cudaEventRecord(stop_e, stream));
				checkCudaErrors(cudaEventSynchronize(stop_e));
				cudaEventElapsedTime(&millis_2, start_e, stop_e);
			}
				
			
			// MoreBreaking
			{
				// Prefetch
				checkCudaErrors(cudaMemPrefetchAsync(candidates, sizeOfCandidates, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(alreadyDeleted, sizeOfAlreadyDeleted, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(output_3, sizeOfOutput, 0, stream));
					
				// Start timer
				checkCudaErrors(cudaEventRecord(start_e, stream));
				// Call kernel
				checkCudaErrors(cudaMemsetAsync(output_3, 0, sizeOfOutput, stream));
				findDublicatesWrapper(ceilf((float)numberOfCandidates/1024),
									  1024,
									  stream,
									  candidates,
									  i, // number of candidates
									  k, // dim
									  alreadyDeleted,
									  output_3,
									  MoreBreaking);
						 
				// End timer
				checkCudaErrors(cudaEventRecord(stop_e, stream));
				checkCudaErrors(cudaEventSynchronize(stop_e));
				cudaEventElapsedTime(&millis_3, start_e, stop_e);
			}


			// Hash
			{
				// Prefetch
				checkCudaErrors(cudaMemPrefetchAsync(candidates, sizeOfCandidates, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(alreadyDeleted, sizeOfAlreadyDeleted, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(output_4, sizeOfOutput, 0, stream));
					
				// Start timer
				checkCudaErrors(cudaEventRecord(start_e, stream));
				// Call kernel
				checkCudaErrors(cudaMemsetAsync(output_4, 0, sizeOfOutput, stream));
				findDublicatesWrapper(ceilf((float)numberOfCandidates/1024),
									  1024,
									  stream,
									  candidates,
									  i, // number of candidates
									  k, // dim
									  alreadyDeleted,
									  output_4,
									  Hash);
						 
				// End timer
				checkCudaErrors(cudaEventRecord(stop_e, stream));
				checkCudaErrors(cudaEventSynchronize(stop_e));
				cudaEventElapsedTime(&millis_4, start_e, stop_e);
			}
				
				
			
			// Prefetch

			checkCudaErrors(cudaMemPrefetchAsync(output_1, sizeOfOutput, cudaCpuDeviceId, stream));
			checkCudaErrors(cudaMemPrefetchAsync(output_2, sizeOfOutput, cudaCpuDeviceId, stream));
			checkCudaErrors(cudaMemPrefetchAsync(output_3, sizeOfOutput, cudaCpuDeviceId, stream));
			checkCudaErrors(cudaMemPrefetchAsync(output_4, sizeOfOutput, cudaCpuDeviceId, stream));
				
			cudaStreamSynchronize(stream); // Synchronize stream 2 to make sure that the data has arrived
			// Check results
			bool passed = true;
			for(unsigned int g = 0; g < 4; g++){
				bool* array;
				if(g == 0){
					array = output_1;
				}else if(g == 1){
					array = output_2;
				}else if(g == 2){
					array = output_3;
				}else if(g == 3){
					array = output_4;
				}

				// for(unsigned int s = 0; s < i; s++){
				// 	std::cout << array[s] << " ";
				// }
				// std::cout << std::endl;
				
				for(unsigned int s = 0; s < duplicates.size() ; s++){
					bool dublicateFound_1 = false;
					for(unsigned int m = 0; m < duplicates.at(s)->size(); m++){
						unsigned int dubIndex = duplicates.at(s)->at(m);
						if(!array[dubIndex]){
							if(dublicateFound_1){
								Experiment::repportError(std::to_string(g) + ": Duplicate present more than once. numberOfCandidates: " + std::to_string(i) + " dim: " + std::to_string(k), this->getName());
								// std::cout << std::to_string(g) + ": Duplicate present more than once. numberOfCandidates: " + std::to_string(i) + " dim: " + std::to_string(k) << std::endl;
								// Dublicates present more than once ERROR
								break;
							}else{
								dublicateFound_1 = true;		
							}
						}
					}
					if(!dublicateFound_1){
						// ERROR not found
						Experiment::repportError(std::to_string(g) + ": Candidate deleted entirely. numberOfCandidates: " + std::to_string(i) + " dim: " + std::to_string(k), this->getName());
					}
				}		
			}
				
			// delete data
			checkCudaErrors(cudaFree(candidates));
			checkCudaErrors(cudaFree(alreadyDeleted));
			checkCudaErrors(cudaFree(output_1));
			checkCudaErrors(cudaFree(output_2));
			checkCudaErrors(cudaFree(output_3));
			checkCudaErrors(cudaFree(output_4));


			// write to file
			Experiment::writeLineToFile(std::to_string(i) + ", " + std::to_string(k) + ", " + "Naive, " + std::to_string(passed) +", " + std::to_string(millis_1));
			Experiment::writeLineToFile(std::to_string(i) + ", " + std::to_string(k) + ", " + "Breaking, " + std::to_string(passed) +", " + std::to_string(millis_2));
			Experiment::writeLineToFile(std::to_string(i) + ", " + std::to_string(k) + ", " + "MoreBreaking, " + std::to_string(passed) +", " + std::to_string(millis_3));
			Experiment::writeLineToFile(std::to_string(i) + ", " + std::to_string(k) + ", " + "Hash, " + std::to_string(passed) +", " + std::to_string(millis_4));

				
			// call test Done
				Experiment::testDone("Dim: " + std::to_string(k) + " numberOfCandidates: " + std::to_string(i));
		}	

	}

	cudaDeviceReset();
	Experiment::stop();
}
