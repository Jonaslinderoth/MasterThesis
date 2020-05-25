#include "mergeCandidatesSmem.h"
#include <random>
#include <math.h>
#include "../src/randomCudaScripts/Utils.h"
#include "../src/MineClusGPU/MergeCandidates.h"

void MergeCandidatesExperimentSmem::start(){
	
	unsigned int c = 0;
	for(unsigned int numberOfCandidates = 32; numberOfCandidates < 1000000; numberOfCandidates*=2){
		for(unsigned int dim = 32; dim < 48000; dim *= 2){
			for(unsigned int chunkSize = 2; chunkSize < 12000; chunkSize *=2){
				unsigned int numberOfNewCandidates = ((float)(numberOfCandidates*(numberOfCandidates+1)) / 2) - numberOfCandidates;
				unsigned int numberOfBlocks = ceilf((float)dim/32);

				size_t sizeOfOutput = numberOfNewCandidates*numberOfBlocks*sizeof(unsigned int);
				size_t sizeOfCandidates = numberOfCandidates*numberOfBlocks*sizeof(unsigned int);
			

				size_t sizeOfToBeDeleted = numberOfNewCandidates*sizeof(bool);

				if((sizeOfOutput+sizeOfCandidates+sizeOfToBeDeleted)*2 > this->memLimit*1000*1000*1000){
					continue;
				}
				unsigned int largestChunkSize = ((48000/4)/2/numberOfBlocks);
				largestChunkSize = log2(largestChunkSize);
				largestChunkSize = pow(2, largestChunkSize);
				if(largestChunkSize <= chunkSize){
					continue;
				}
				c++;
			}
		}
	}
	
	Experiment::addTests(c);
	Experiment::start();

	for(unsigned int numberOfCandidates = 32; numberOfCandidates < 1000000; numberOfCandidates*=2){
		for(unsigned int dim = 32; dim < 48000; dim *= 2){
			for(unsigned int chunkSize = 2; chunkSize < 12000; chunkSize *=2){
				unsigned int numberOfNewCandidates = ((float)(numberOfCandidates*(numberOfCandidates+1)) / 2) - numberOfCandidates;
				unsigned int numberOfBlocks = ceilf((float)dim/32);

				size_t sizeOfOutput = numberOfNewCandidates*numberOfBlocks*sizeof(unsigned int);
				size_t sizeOfCandidates = numberOfCandidates*numberOfBlocks*sizeof(unsigned int);
			
				size_t sizeOfToBeDeleted = numberOfNewCandidates*sizeof(bool);

				if((sizeOfOutput+sizeOfCandidates+sizeOfToBeDeleted)*2 > this->memLimit*1000*1000*1000){
					continue;
				}
				unsigned int largestChunkSize = ((48000/4)/2/numberOfBlocks);
				largestChunkSize = log2(largestChunkSize);
				largestChunkSize = pow(2, largestChunkSize);
				if(largestChunkSize <= chunkSize){
					continue;
				}
				
				unsigned int dimBlock = 1024;
				unsigned int dimGrid = ceilf((float)numberOfNewCandidates/dimBlock);
				dimGrid = dimGrid == 0 ? 1 : dimGrid;

				unsigned int* candidates_h;

				cudaMallocHost((void**) &candidates_h, sizeOfCandidates);

				std::mt19937 rng(std::random_device{}());
				rng.seed(0);
			
				std::uniform_int_distribution<> distr1(0, dim);

				for(unsigned int i = 0; i < numberOfCandidates; i++){
					unsigned int block = 0;
					unsigned int blockNr = 0;
					for(unsigned int j = 0; j < dim/32; j++){
						block = 0;
						candidates_h[(size_t)i+j*numberOfCandidates] = block;
					}
				}
				for(unsigned int i = 0; i < numberOfCandidates; i++){
					unsigned int block = 0;
					unsigned int blockNr = 0;
					for(unsigned int j = 0; j < 4; j++){
						auto tmp = distr1(rng);
						block = candidates_h[(size_t)i+(tmp/32)*numberOfCandidates];
						block |= (1 << tmp%32);
						candidates_h[(size_t)i+(tmp/32)*numberOfCandidates] = block;
					}
				}



				cudaStream_t stream;
				checkCudaErrors(cudaStreamCreate(&stream));
				cudaEvent_t start_e, stop_e;
				cudaEventCreate(&start_e);
				cudaEventCreate(&stop_e);

				float millisSmem = 0;
				unsigned int k = 4;

		

			
				// Smem
				{			
					unsigned int* candidates_d;
					unsigned int* output_d;
					bool* toBeDeleted_d;
					cudaMalloc((void**) &candidates_d, sizeOfCandidates);
					cudaMalloc((void**) &output_d, sizeOfOutput);
					cudaMalloc((void**) &toBeDeleted_d, sizeOfToBeDeleted);
			
					cudaMemcpy(candidates_d, candidates_h, sizeOfCandidates, cudaMemcpyHostToDevice);
					checkCudaErrors(cudaEventRecord(start_e, stream));
					mergeCandidatesWrapper(dimGrid, dimBlock, stream, candidates_d, numberOfCandidates, dim, k, output_d, toBeDeleted_d, SharedMemoryMerge, chunkSize);
					checkCudaErrors(cudaEventRecord(stop_e, stream));

					checkCudaErrors(cudaEventSynchronize(stop_e));
					cudaEventElapsedTime(&millisSmem, start_e, stop_e);
				
					cudaFree(candidates_d);
					cudaFree(output_d);
					cudaFree(toBeDeleted_d);
				}


						
				Experiment::writeLineToFile(std::to_string(numberOfCandidates) + ", " + std::to_string(dim) + ", " +  std::to_string(chunkSize) + ", " +  std::to_string(millisSmem));

				if(numberOfCandidates > 2){;
					cudaFreeHost(candidates_h);
				}

				Experiment::testDone("Dim: " + std::to_string(dim) + " numPoints: " + std::to_string(numberOfCandidates));
				checkCudaErrors(cudaStreamDestroy(stream));
			}
		}
	}



	cudaDeviceReset();
	Experiment::stop();
}
