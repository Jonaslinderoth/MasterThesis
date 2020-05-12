#include "createTransactions.h"
#include <random>
#include "../src/MineClusGPU/MineClusKernels.h"
#include "../src/MineClusGPU/CreateTransactions.h"
#include "../src/randomCudaScripts/Utils.h"
void CreateTransactionsExperiments::start(){
	std::random_device rd;
	std::mt19937 gen(rd());
	gen.seed(0); // important to seed
	std::uniform_real_distribution<> dist(-100, 100);
	unsigned int c = 0;
	for(unsigned int numPoints = 2; numPoints < 100000; numPoints*=2){
		for(unsigned int dim = 2; dim < 100000; dim *= 2){
			c++;
		}
	}
	
	Experiment::addTests(c);
	Experiment::start();
	for(unsigned int numPoints = 2; numPoints < 100000; numPoints*=2){
		for(unsigned int dim = 2; dim < 100000; dim *= 2){
			// create the initial stuff
			size_t size = numPoints;
			size_t size_of_data = size*dim*sizeof(float);
			size_t size_of_output = size*ceilf((float)dim/32)*sizeof(unsigned int);
			float* data_h;
			checkCudaErrors(cudaMallocHost((void**) &data_h, size_of_data));
			// create data, just do random data
			for(size_t i = 0; i < numPoints; i++){
				for(size_t j = 0; j < dim; j++){
					data_h[(size_t)i*dim+j] = dist(gen);
				}
			}
			unsigned int centroid = numPoints/2;
			float width = 15;
			std::vector<unsigned int> resNaive;
			std::vector<unsigned int> resReducedReads;
			cudaStream_t stream;
			checkCudaErrors(cudaStreamCreate(&stream));
			cudaEvent_t start_e, stop_e;
			cudaEventCreate(&start_e);
			cudaEventCreate(&stop_e);
			float millisNaive = 0;
			float millisReducedReads = 0;
			// Naive
			{
				float* data_d;
				checkCudaErrors(cudaMalloc((void**) &data_d, size_of_data));
				checkCudaErrors(cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice));

				unsigned int* output_d;
				checkCudaErrors(cudaMalloc((void**) &output_d, size_of_output));
				unsigned int* output_h;
				checkCudaErrors(cudaMallocHost((void**) &output_h, size_of_output));
				
				unsigned int dimBlock = 1024;
				unsigned int dimGrid = ceilf((float)size/dimBlock);
				checkCudaErrors(cudaEventRecord(start_e, stream));
				createTransactionsWrapper(dimGrid, dimBlock,0, stream, data_d, dim, size, centroid, width, output_d, Naive_trans);
				checkCudaErrors(cudaEventRecord(stop_e, stream));
				checkCudaErrors(cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost));

				checkCudaErrors(cudaEventSynchronize(stop_e));
				cudaEventElapsedTime(&millisNaive, start_e, stop_e);
				
				checkCudaErrors(cudaFree(data_d));
				checkCudaErrors(cudaFree(output_d));
			
				for(unsigned int i = 0; i < ceilf((float)dim/32)*size;i++){
					resNaive.push_back(output_h[i]);
				}
				checkCudaErrors(cudaFreeHost(output_h));
			}

			// Reduced reads
			{
				float* data_d;
				checkCudaErrors(cudaMalloc((void**) &data_d, size_of_data));
				checkCudaErrors(cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice));
				unsigned int* output_d;
				checkCudaErrors(cudaMalloc((void**) &output_d, size_of_output));
				unsigned int* output_h;
				checkCudaErrors(cudaMallocHost((void**) &output_h, size_of_output));
				unsigned int dimBlock = 1024;
				unsigned int dimGrid = ceilf((float)size/dimBlock);
				unsigned int smem_size = 48000;
				checkCudaErrors(cudaEventRecord(start_e, stream));
				createTransactionsWrapper(dimGrid, dimBlock, smem_size, stream, data_d, dim, size, centroid, width, output_d, ReducedReads);
				checkCudaErrors(cudaEventRecord(stop_e, stream));
				
				checkCudaErrors(cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost));

				cudaEventSynchronize(stop_e);
				cudaEventElapsedTime(&millisReducedReads, start_e, stop_e);

				checkCudaErrors(cudaFree(data_d));
				checkCudaErrors(cudaFree(output_d));

				for(size_t i = 0; i < (unsigned int)ceilf((float)dim/32)*size;i++){
					resReducedReads.push_back(output_h[i]);
				}
			}

			cudaFreeHost(data_h);
			checkCudaErrors(cudaStreamDestroy(stream));
			bool passed = true;
			// compare results
			if(resReducedReads.size() != resNaive.size()){
				Experiment::repportError("Sizes do not match: " + std::to_string(resReducedReads.size()) + " vs " + std::to_string(resNaive.size()), this->getName());
				passed = false;
				break;
			}
			for(size_t i = 0; i < resReducedReads.size(); i++){
				if(resReducedReads.at(i) != resNaive.at(i)){
					Experiment::repportError("Results are not the same, at index: " + std::to_string(i) + " : " + std::to_string(resReducedReads.at(i)) + " vs " + std::to_string(resNaive.at(i)), this->getName());
					passed = false;
					break;
				}
			}
			
			
			Experiment::writeLineToFile(std::to_string(numPoints) + ", " + std::to_string(dim) + ", " + "Naive, " + std::to_string(passed) +", " + std::to_string(millisNaive));
			Experiment::writeLineToFile(std::to_string(numPoints) + ", " + std::to_string(dim) + ", " + "ReducedReads, " + std::to_string(passed) +", " + std::to_string(millisReducedReads));
			Experiment::testDone("Dim: " + std::to_string(dim) + " numPoints: " + std::to_string(numPoints));
			
		}
	}
	cudaDeviceReset();
	Experiment::stop();
}
