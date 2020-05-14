#include "ExperimentCountSupport.h"
#include <random>
#include "../src/MineClusGPU/CountSupport.h"
#include "../src/randomCudaScripts/Utils.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

void ExperimentCountSupport::start(){
	unsigned int c = 0;
	unsigned int numberOfCandidates = 30000;
	unsigned int numberOfPoints =     30000;
	unsigned int dim =                1024;

	// count tests
	for(unsigned int i = 1024; i < numberOfCandidates; i += 1024*2){
		for(unsigned int j = 1024; j < numberOfPoints; j += 1024*2){
			for(unsigned int k = 32; k < dim; k+= 32){
				c++;
			}	
		}		
	}

	std::mt19937 rng(std::random_device{}());
	rng.seed(0);;
	Experiment::addTests(c);
	Experiment::start();

	
	// Do experiments
	for(unsigned int i = 1024; i < numberOfCandidates; i += 1024*2){
		for(unsigned int j = 1024; j < numberOfPoints; j += 1024*2){
			for(unsigned int k = 32; k < dim; k+= 32){
				// Calculate sizes
				size_t sizeOfTransactions = j*(k/32)*sizeof(unsigned int);
				size_t sizeOfCandidates = i*(k/32)*sizeof(unsigned int);
				size_t sizeOfScore = i*sizeof(float);
				size_t sizeOfCounts = i*sizeof(unsigned int);
				size_t sizeOfToBeDeleted = i*sizeof(bool);

				unsigned int* transactions;
				unsigned int* candidates;
				float* score;
				unsigned int* counts;
				bool* toBeDeleted; 
				float* score_2;
				unsigned int* counts_2;
				bool* toBeDeleted_2; 				
				// Allocate unified memory
				checkCudaErrors(cudaMallocManaged((void**) &transactions, sizeOfTransactions));
				checkCudaErrors(cudaMallocManaged((void**) &candidates, sizeOfCandidates));
				checkCudaErrors(cudaMallocManaged((void**) &score, sizeOfScore));
				checkCudaErrors(cudaMallocManaged((void**) &counts, sizeOfCounts));
				checkCudaErrors(cudaMallocManaged((void**) &toBeDeleted, sizeOfToBeDeleted));
				
				checkCudaErrors(cudaMallocManaged((void**) &score_2, sizeOfScore));
				checkCudaErrors(cudaMallocManaged((void**) &counts_2, sizeOfCounts));
				checkCudaErrors(cudaMallocManaged((void**) &toBeDeleted_2, sizeOfToBeDeleted));
				
				// Generate transactions
				for(unsigned int jj = 0; jj < j; jj++){
					for(unsigned int kk = 0; kk < k/32; kk++){
						unsigned int block = 0;
						for(unsigned int blockIndex = 0; blockIndex < 32; blockIndex++){
							// jj is to make an offset
							bool value = (((jj + kk*32+blockIndex)%10) == 0);
							if(value){
								block |= (1 << blockIndex);
							}
						}
						transactions[jj+ kk*j] = block;
					}
				}
				// Generate candidates
				for(unsigned int ii = 0; ii < i; ii++){
					for(unsigned int kk = 0; kk < k/32; kk++){
						unsigned int block = 0;
						for(unsigned int blockIndex = 0; blockIndex < 32; blockIndex++){
							// jj is to make an offset
							bool value = (kk*32+blockIndex) == ii;
							if(value){
								block |= (1 << blockIndex);
							}
							if((kk*32+blockIndex) > k){
								auto tmp = ((std::uniform_int_distribution<int>{}(rng)) & 1);
								block |= (tmp << blockIndex);
							}
						}
						//if(block != 0) Experiment::repportError(std::to_string(block), this->getName());
						candidates[ii+ kk*i] = block;
					}				
				}

				cudaStream_t stream;
				checkCudaErrors(cudaStreamCreate(&stream));
				cudaEvent_t start_e, stop_e;
				cudaEventCreate(&start_e);
				cudaEventCreate(&stop_e);
				float millisNaive = 0;
				float millisSmem = 0;

				// Naive
				{
					// Prefetch
					checkCudaErrors(cudaMemPrefetchAsync(transactions, sizeOfTransactions, 0, stream));
					checkCudaErrors(cudaMemPrefetchAsync(candidates, sizeOfCandidates, 0, stream));
					checkCudaErrors(cudaMemPrefetchAsync(score, sizeOfScore, 0, stream));
					checkCudaErrors(cudaMemPrefetchAsync(counts, sizeOfCounts, 0, stream));
					checkCudaErrors(cudaMemPrefetchAsync(toBeDeleted, sizeOfToBeDeleted, 0, stream));
					
					// Start timer
					checkCudaErrors(cudaEventRecord(start_e, stream));
					// Call kernel
					countSupportWrapper(ceilf((float)numberOfCandidates/1024),
										1024,
										stream,
										candidates,
										transactions,
										k, // dim
										j, // number of point
										i, // number of candidates
										1,
										0.25,
										counts,
										score,
										toBeDeleted,
										NaiveCount);
						 
					// End timer
					checkCudaErrors(cudaEventRecord(stop_e, stream));
					checkCudaErrors(cudaEventSynchronize(stop_e));
					cudaEventElapsedTime(&millisNaive, start_e, stop_e);
				}
				
				// Shared
				{
					// Prefetch
					checkCudaErrors(cudaMemPrefetchAsync(transactions, sizeOfTransactions, 0, stream));
					checkCudaErrors(cudaMemPrefetchAsync(candidates, sizeOfCandidates, 0, stream));
					checkCudaErrors(cudaMemPrefetchAsync(score_2, sizeOfScore, 0, stream));
					checkCudaErrors(cudaMemPrefetchAsync(counts_2, sizeOfCounts, 0, stream));
					checkCudaErrors(cudaMemPrefetchAsync(toBeDeleted_2, sizeOfToBeDeleted, 0, stream));
					
					// Start timer
					checkCudaErrors(cudaEventRecord(start_e, stream));
					// Call kernel
					countSupportWrapper(ceilf((float)numberOfCandidates/1024),
										1024,
										stream,
										candidates,
										transactions,
										k,
										j,
										i,
										1,
										0.25,
										counts_2,
										score_2,
										toBeDeleted_2,
										SmemCount);
					// End timer
					checkCudaErrors(cudaEventRecord(stop_e, stream));
					checkCudaErrors(cudaEventSynchronize(stop_e));
					cudaEventElapsedTime(&millisSmem, start_e, stop_e);
				}



				// Prefetch
				checkCudaErrors(cudaMemPrefetchAsync(score_2, sizeOfScore, cudaCpuDeviceId, stream));
				checkCudaErrors(cudaMemPrefetchAsync(counts_2, sizeOfCounts, cudaCpuDeviceId, stream));
				checkCudaErrors(cudaMemPrefetchAsync(toBeDeleted_2, sizeOfToBeDeleted, cudaCpuDeviceId, stream));
				checkCudaErrors(cudaMemPrefetchAsync(score, sizeOfScore, cudaCpuDeviceId, stream));
				checkCudaErrors(cudaMemPrefetchAsync(counts, sizeOfCounts, cudaCpuDeviceId, stream));
				checkCudaErrors(cudaMemPrefetchAsync(toBeDeleted, sizeOfToBeDeleted, cudaCpuDeviceId, stream));
				cudaStreamSynchronize(stream); // Synchronize stream 2 to make sure that the data has arrived
				// Check results
				bool passed = true;
				for(unsigned int s = 0; s < i; s++){
					if(counts[s] != counts_2[s]){
						Experiment::repportError("Counts do not match " + std::to_string(counts[s]) +" != " + std::to_string(counts_2[s]) + " i: " +std::to_string(i)+ " j: "+ std::to_string(j)+ " k: "+ std::to_string(k), this->getName());
						passed = false;
					}
					if(score[s] != score_2[s]){
						Experiment::repportError("Scores do not match " + std::to_string(score[s]) +" != " + std::to_string(score_2[s]) + " i: "+ std::to_string(i)+ " j: "+ std::to_string(j)+ " k: "+ std::to_string(k) + ", s: " +std::to_string(s), this->getName());
						// std::cout << "Scores do not match " + std::to_string(score[s]) +" != " + std::to_string(score_2[s]) + " i: "+ std::to_string(i)+ " j: "+ std::to_string(j)+ " k: "+ std::to_string(k) + ", s: " +std::to_string(s) << std::endl;
						passed = false;
					}
					if(toBeDeleted[s] != toBeDeleted_2[s]){
						Experiment::repportError("toBeDeleted do not match " + std::to_string(toBeDeleted[s]) +" != " + std::to_string(toBeDeleted_2[s]) + " i: "+ std::to_string(i)+ " j: " +std::to_string(j)+ " k: " +std::to_string(k), this->getName());
						passed = false;
						
					}
				}
				
				// delete data
				checkCudaErrors(cudaFree(transactions));
				checkCudaErrors(cudaFree(candidates));
				checkCudaErrors(cudaFree(score));
				checkCudaErrors(cudaFree(counts));
				checkCudaErrors(cudaFree(toBeDeleted));
				checkCudaErrors(cudaFree(score_2));
				checkCudaErrors(cudaFree(counts_2));
				checkCudaErrors(cudaFree(toBeDeleted_2));

				// write to file
				Experiment::writeLineToFile(std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ", " + "Naive, " + std::to_string(passed) +", " + std::to_string(millisNaive));
				Experiment::writeLineToFile(std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + ", " + "Shared, " + std::to_string(passed) +", " + std::to_string(millisSmem));
				
				// call test Done
				Experiment::testDone("Dim: " + std::to_string(k) + " numberOfCandidates: " + std::to_string(i) + " numPoints: " + std::to_string(j));
			}	
		}		
	}

	cudaDeviceReset();
	Experiment::stop();
}
