#include "ExperimentFindDimensions.h"
#include "../src/DOC_GPU/DOCGPU_Kernels.h"
#include "../src/randomCudaScripts/Utils.h"
#include "../src/testingTools/DataGeneratorBuilder.h"
#include "../src/dataReader/DataReader.h"
#include <random>

void ExperimentFindDimensions::start(){
	std::mt19937 gen;
	unsigned int dim = 1024;
	unsigned int numberOfPoints = 16384;
	unsigned int c = 0;

	for(unsigned int i = 8; i <= dim; i*=2){
		c++;
		c++;
	}
	Experiment::addTests(c);
	Experiment::start();
	for(unsigned int i = 8; i <= dim; i*=2){
		{
			unsigned int j = numberOfPoints;
			unsigned int sampleSize = log2(2*i)/log2(1/(2*0.25));
			size_t sizeOfMedoids = 19*sizeof(unsigned int);
			size_t sizeOfSamples = 1000000*sampleSize*19*sizeof(unsigned int);
			size_t sizeOfData = i*j*sizeof(float);
			size_t sizeOfOutput = 1000000*19*i*sizeof(bool);
			size_t sizeOfOutputCount = 1000000*19*sizeof(unsigned int);

			float* data;
			unsigned int* medoids;
			unsigned int* samples;
			bool* output1;
			bool* output2;
			unsigned int* output1Count;
			unsigned int* output2Count;
			
			checkCudaErrors(cudaMallocManaged((void**) &data, sizeOfData));
			checkCudaErrors(cudaMallocManaged((void**) &medoids, sizeOfMedoids));
			checkCudaErrors(cudaMallocManaged((void**) &samples, sizeOfSamples));
			checkCudaErrors(cudaMallocManaged((void**) &output1, sizeOfOutput));
			checkCudaErrors(cudaMallocManaged((void**) &output2, sizeOfOutput));
			checkCudaErrors(cudaMallocManaged((void**) &output1Count, sizeOfOutputCount));
			checkCudaErrors(cudaMallocManaged((void**) &output2Count, sizeOfOutputCount));

			std::uniform_int_distribution<> dis(0, j-1);
			for(unsigned int n = 0; n < 19; n++){
				medoids[n] = dis(gen);
			}
			for(unsigned int n = 0; n < 1000000*sampleSize*19; n++){
				samples[n] = dis(gen);
			}

			// Generate data u-clusters
			// read from data reader
			// paste it into the array
			if(system("mkdir testData  >>/dev/null 2>>/dev/null")){
				
			};
			DataGeneratorBuilder dgb;
			dgb.setSeed(rand());
			bool res = dgb.buildUClusters("testData/test1",j/8,8,15,i,5,0, true);
			DataReader* dr = new DataReader("testData/test1");
			auto size = j;
			std::vector<std::vector<float>*>* data2 = new std::vector<std::vector<float>*>(0);
			data2->reserve(size);

			while(dr->isThereANextBlock()){
				std::vector<std::vector<float>*>* block = dr->next();
				data2->insert(data2->end(), block->begin(), block->end());
				delete block;
			}
			for(unsigned int ii = 0; ii < j; ii++){
				for(unsigned int jj = 0; jj < i; jj++){
			
					data[(size_t)ii*i+jj] = data2->at(ii)->at(jj);
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
				checkCudaErrors(cudaMemPrefetchAsync(data, sizeOfData, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(medoids, sizeOfMedoids, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(samples, sizeOfSamples, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(output1, sizeOfOutput, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(output1Count, sizeOfOutputCount, 0, stream));
				checkCudaErrors(cudaEventRecord(start_e, stream));

				findDimmensionsKernel(ceilf((float)j/1024), 1024, stream,
									  samples, medoids, data, output1,
									  output1Count, i,
									  1000000*19, sampleSize,
									  19,
									  1000000, 15, j,
									  naiveFindDim);


				
				checkCudaErrors(cudaEventRecord(stop_e, stream));
				checkCudaErrors(cudaEventSynchronize(stop_e));
				cudaEventElapsedTime(&millisNaive, start_e, stop_e);
				
			}
			Experiment::testDone("Naive Dim: " + std::to_string(i) + " numberOfPoints" + std::to_string(j));

			// chunks
			{
				checkCudaErrors(cudaMemPrefetchAsync(data, sizeOfData, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(medoids, sizeOfMedoids, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(samples, sizeOfSamples, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(output2, sizeOfOutput, 0, stream));
				checkCudaErrors(cudaMemPrefetchAsync(output2Count, sizeOfOutputCount, 0, stream));
				checkCudaErrors(cudaEventRecord(start_e, stream));

				findDimmensionsKernel(ceilf((float)j/1024), 1024, stream,
									  samples, medoids, data, output2,
									  output2Count, i,
									  1000000*19, sampleSize,
									  19,
									  1000000, 15, j,
									  chunksFindDim);

				
				checkCudaErrors(cudaEventRecord(stop_e, stream));
				checkCudaErrors(cudaEventSynchronize(stop_e));
				cudaEventElapsedTime(&millisSmem, start_e, stop_e);
				
			}
			Experiment::testDone("Chunks Dim: " + std::to_string(i) + " numberOfPoints" + std::to_string(j));
			
			checkCudaErrors(cudaMemPrefetchAsync(output1, sizeOfOutput, cudaCpuDeviceId, stream));
			checkCudaErrors(cudaMemPrefetchAsync(output2, sizeOfOutput, cudaCpuDeviceId, stream));
			checkCudaErrors(cudaMemPrefetchAsync(output1Count, sizeOfOutputCount, cudaCpuDeviceId, stream));
			checkCudaErrors(cudaMemPrefetchAsync(output2Count, sizeOfOutputCount, cudaCpuDeviceId, stream));


			bool eq = true;
			for(unsigned int n = 0; n < 1000000*19*i; n++){
				eq &= output1[n] == output2[n];
				if(!eq){
					Experiment::repportError ("Outputs do not match", this->getName());
					break;
				}
			}
			bool eq2=true;
			for(unsigned int n = 0; n < 1000000*19; n++){
				eq2 &= output1Count[n] == output2Count[n];
				if(!eq2){
					Experiment::repportError("Counts do not match " + std::to_string(output1Count[n]) + " vs " + std::to_string(output2Count[n]), this->getName());
				}
			}

			cudaFree(data);
			cudaFree(medoids);
			cudaFree(samples);
			cudaFree(output1);
			cudaFree(output2);
			cudaFree(output1Count);
			cudaFree(output2Count);
			
			
			Experiment::writeLineToFile(std::to_string(j) + ", " + std::to_string(i) + ", Naive, " + std::to_string(eq && eq2) + ", " + std::to_string(millisNaive));
										
			Experiment::writeLineToFile(std::to_string(j) + ", " + std::to_string(i) + ", Chunks, " + std::to_string(eq && eq2) + ", " + std::to_string(millisSmem));
			// prefetch
			// call kernel
			
			// compare results

			
			
			
			
		}
	}

	Experiment::stop();
}
