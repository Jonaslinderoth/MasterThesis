#include "mergeCandidates.h"
#include <random>
#include "../src/randomCudaScripts/Utils.h"
#include "../src/MineClusGPU/MergeCandidates.h"

void MergeCandidatesExperiment::start(){
	
	unsigned int c = 0;
	for(unsigned int numberOfCandidates = 2; numberOfCandidates < 1000000; numberOfCandidates*=2){
		for(unsigned int dim = 32; dim < 48000; dim *= 2){
			unsigned int numberOfNewCandidates = ((float)(numberOfCandidates*(numberOfCandidates+1)) / 2) - numberOfCandidates;
			unsigned int numberOfBlocks = ceilf((float)dim/32);

			size_t sizeOfOutput = numberOfNewCandidates*numberOfBlocks*sizeof(unsigned int);
			size_t sizeOfCandidates = numberOfCandidates*numberOfBlocks*sizeof(unsigned int);
			

			size_t sizeOfToBeDeleted = numberOfNewCandidates*sizeof(bool);

			if((sizeOfOutput+sizeOfCandidates+sizeOfToBeDeleted)*3 > this->memLimit*1000*1000*1000){
				continue;
			}
			c++;
		}
	}
	
	Experiment::addTests(c);
	Experiment::start();

	for(unsigned int numberOfCandidates = 2; numberOfCandidates < 1000000; numberOfCandidates*=2){
		for(unsigned int dim = 32; dim < 48000; dim *= 2){
			unsigned int numberOfNewCandidates = ((float)(numberOfCandidates*(numberOfCandidates+1)) / 2) - numberOfCandidates;
			unsigned int numberOfBlocks = ceilf((float)dim/32);

			size_t sizeOfOutput = numberOfNewCandidates*numberOfBlocks*sizeof(unsigned int);
			size_t sizeOfCandidates = numberOfCandidates*numberOfBlocks*sizeof(unsigned int);
			

			size_t sizeOfToBeDeleted = numberOfNewCandidates*sizeof(bool);

			if((sizeOfOutput+sizeOfCandidates+sizeOfToBeDeleted)*3 > this->memLimit*1000*1000*1000){
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

			

				unsigned int* output_naive_h;
				bool* toBeDeleted_naive_h;
				unsigned int* output_early_h;
				bool* toBeDeleted_early_h;
				unsigned int* output_smem_h;
				bool* toBeDeleted_smem_h;
				cudaMallocHost((void**) &output_naive_h, sizeOfOutput);
				cudaMallocHost((void**) &toBeDeleted_naive_h, sizeOfToBeDeleted);
				cudaMallocHost((void**) &output_early_h, sizeOfOutput);
				cudaMallocHost((void**) &toBeDeleted_early_h, sizeOfToBeDeleted);
				cudaMallocHost((void**) &output_smem_h, sizeOfOutput);
				cudaMallocHost((void**) &toBeDeleted_smem_h, sizeOfToBeDeleted);
				cudaStream_t stream;
				checkCudaErrors(cudaStreamCreate(&stream));
				cudaEvent_t start_e, stop_e;
				cudaEventCreate(&start_e);
				cudaEventCreate(&stop_e);

				float millisNaive = 0;
				float millisEarly = 0;
				float millisSmem = 0;
				unsigned int k = 4;

				// naive
			{			
				unsigned int* candidates_d;
				unsigned int* output_d;
				bool* toBeDeleted_d;
				cudaMalloc((void**) &candidates_d, sizeOfCandidates);
				cudaMalloc((void**) &output_d, sizeOfOutput);
				cudaMalloc((void**) &toBeDeleted_d, sizeOfToBeDeleted);
			
				cudaMemcpy(candidates_d, candidates_h, sizeOfCandidates, cudaMemcpyHostToDevice);
				checkCudaErrors(cudaEventRecord(start_e, stream));
				mergeCandidatesWrapper(dimGrid, dimBlock, stream, candidates_d, numberOfCandidates, dim, k, output_d, toBeDeleted_d, NaiveMerge);
				checkCudaErrors(cudaEventRecord(stop_e, stream));

				cudaMemcpy(output_naive_h, output_d, sizeOfOutput, cudaMemcpyDeviceToHost);
				cudaMemcpy(toBeDeleted_naive_h, toBeDeleted_d, sizeOfToBeDeleted, cudaMemcpyDeviceToHost);
				checkCudaErrors(cudaEventSynchronize(stop_e));
				cudaEventElapsedTime(&millisNaive, start_e, stop_e);

				cudaFree(candidates_d);
				cudaFree(output_d);
				cudaFree(toBeDeleted_d);
			}

			// Early
			{			
				unsigned int* candidates_d;
				unsigned int* output_d;
				bool* toBeDeleted_d;
				cudaMalloc((void**) &candidates_d, sizeOfCandidates);
				cudaMalloc((void**) &output_d, sizeOfOutput);
				cudaMalloc((void**) &toBeDeleted_d, sizeOfToBeDeleted);
			
				cudaMemcpy(candidates_d, candidates_h, sizeOfCandidates, cudaMemcpyHostToDevice);
				checkCudaErrors(cudaEventRecord(start_e, stream));
				mergeCandidatesWrapper(dimGrid, dimBlock, stream, candidates_d, numberOfCandidates, dim, k, output_d, toBeDeleted_d, EarlyStoppingMerge);
				checkCudaErrors(cudaEventRecord(stop_e, stream));

				cudaMemcpy(output_early_h, output_d, sizeOfOutput, cudaMemcpyDeviceToHost);
				cudaMemcpy(toBeDeleted_early_h, toBeDeleted_d, sizeOfToBeDeleted, cudaMemcpyDeviceToHost);
				checkCudaErrors(cudaEventSynchronize(stop_e));
				cudaEventElapsedTime(&millisEarly, start_e, stop_e);

				cudaFree(candidates_d);
				cudaFree(output_d);
				cudaFree(toBeDeleted_d);
			}

			
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
				mergeCandidatesWrapper(dimGrid, dimBlock, stream, candidates_d, numberOfCandidates, dim, k, output_d, toBeDeleted_d, SharedMemoryMerge);
				checkCudaErrors(cudaEventRecord(stop_e, stream));

				cudaMemcpy(output_smem_h, output_d, sizeOfOutput, cudaMemcpyDeviceToHost);
				cudaMemcpy(toBeDeleted_smem_h, toBeDeleted_d, sizeOfToBeDeleted, cudaMemcpyDeviceToHost);
				checkCudaErrors(cudaEventSynchronize(stop_e));
				cudaEventElapsedTime(&millisSmem, start_e, stop_e);
				
				cudaFree(candidates_d);
				cudaFree(output_d);
				cudaFree(toBeDeleted_d);
			}



				bool passed = true;
				unsigned int trues =0;
				unsigned int falses = 0;
				for(unsigned int i = 0; i < numberOfNewCandidates; i++){
					if(toBeDeleted_naive_h[i]){
						trues++;
					}else{
						falses++;
					}
					if(toBeDeleted_naive_h[i] == toBeDeleted_early_h[i] && toBeDeleted_early_h[i] == toBeDeleted_smem_h[i]){
						
					}else{
						passed = false;
						Experiment::repportError("toBeDeleted do not match", this->getName());
						break; 
					}
				}

				
				bool passed2 = true;
				for(unsigned int i = 0; i < numberOfNewCandidates*numberOfBlocks; i++){
					if(output_naive_h[i] == output_smem_h[i]){
						
					}else{
						passed = false;
						Experiment::repportError("output do not match", this->getName());
						break; 
					}
				}

				if(passed && passed2){
					Experiment::writeLineToFile(std::to_string(numberOfCandidates) + ", " + std::to_string(dim) + ", " + "Naive, " + std::to_string(passed && passed2) +", " + std::to_string(millisNaive) + ", " + std::to_string(trues));
					Experiment::writeLineToFile(std::to_string(numberOfCandidates) + ", " + std::to_string(dim) + ", " + "Early, " + std::to_string(passed && passed2) +", " + std::to_string(millisEarly) + ", " + std::to_string(trues));
					Experiment::writeLineToFile(std::to_string(numberOfCandidates) + ", " + std::to_string(dim) + ", " + "SharedMemory, " + std::to_string(passed && passed2) +", " + std::to_string(millisSmem) + ", " + std::to_string(trues));
				}

				if(numberOfCandidates > 2){
					cudaFreeHost(toBeDeleted_naive_h);
					cudaFreeHost(toBeDeleted_early_h);
					cudaFreeHost(toBeDeleted_smem_h);
					cudaFreeHost(output_naive_h);
					cudaFreeHost(output_early_h);
					cudaFreeHost(output_smem_h);	
				}

				Experiment::testDone("Dim: " + std::to_string(dim) + " numPoints: " + std::to_string(numberOfCandidates));
				checkCudaErrors(cudaStreamDestroy(stream));
		}
	}



	cudaDeviceReset();
	Experiment::stop();
}
