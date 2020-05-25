#include "ExperimentClusteringSpeedLarge.h"
#include <unistd.h>
#include <iostream>
#include "../src/testingTools/DataGeneratorBuilder.h"
#include "../src/dataReader/DataReader.h"
#include "../src/DOC/DOC.h"
#include "../src/DOC_GPU/DOCGPU.h"
#include "../src/DOC_GPU/DOCGPUnified.h"
#include "../src/Fast_DOC/Fast_DOC.h"
#include "../src/Fast_DOCGPU/Fast_DOCGPU.h"
#include "../src/Fast_DOCGPU/Fast_DOCGPUnified.h"
#include "../src/MineClus/MineClus.h"
#include "../src/MineClusGPU/MineClusGPU.h"
#include "../src/MineClusGPU/MineClusGPUnified.h"
#include "../src/Evaluation.h"
#include <random>
#include <chrono>

void ExperimentClusteringSpeedLarge::start(){
	unsigned int dim = 2000;
	unsigned int numberOfPointsPerCluster = 1000000;
	unsigned int usedDim = 5;
	unsigned int c = 0; 
	// Count number of tests
	for(unsigned int i = 16; i <= dim; i *= 2){
		for(unsigned int j = 32; j <= numberOfPointsPerCluster; j *=2){
			for(unsigned int k = 5; k <= usedDim; k +=10){
				if (k*2 > i) break;
				c+=13;
			}
		}
	}

	Experiment::addTests(c);
	
	Experiment::start();
	int seed = 0;
		// for(unsigned int i = 16; i <= dim; i *= 2)
	{
		i = dim;
		//for(unsigned int j = 32; j <= numberOfPointsPerCluster; j *=2)
		{
			j = numberOfPointsPerCluster;
			for(unsigned int k = 5; k <= usedDim; k +=10){
				if (k*2 > i) break;
				// create dataBuilder
				DataGeneratorBuilder dgb;
				dgb.setSeed(0);
				if(system("mkdir testData  >>/dev/null 2>>/dev/null")){
					
				};
				bool res = dgb.buildUClusters("testData/test1",j,10,15,i,k,5, true);

				if(!res){this->repportError("Cluster not generated", this->getName());
				}else{
					Experiment::testDone("Cluster generated. Number of points: " + std::to_string(j*10) + " Dims used: " + std::to_string(k) + "Dim " + std::to_string(i));
				}
				seed = i*k*j;
				/******************************** DOC TESTS ********************************/
	

				try{
					if(true){
						DataReader* dr = new DataReader("testData/test1");
						DOCGPUnified c = DOCGPUnified(dr);
						c.setWidth(15);
						c.setSeed(seed);
						c.setAlpha(0.1);
						c.setBeta(0.25);
						c.setFindDimVersion(chunksFindDim);
						// c.setPointsContainedVersion(pointContainedSmem);
						c.setNumberOfSamples(4096*2);

						// start timer
						auto t1 = std::chrono::high_resolution_clock::now();
						auto result = c.findKClusters(10);
						auto t2 = std::chrono::high_resolution_clock::now();
						// stop timer
						auto time = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
						float acc = 0;
						if(j <= 2048){
							auto confusion = Evaluation::confusion(labels, result);
							acc = Evaluation::accuracy(confusion);
						}

						
						this->writeLineToFile(std::to_string(10) + ", "
											  + std::to_string(j*10) + ", "
											  +  std::to_string(i) + ", "
											  + std::to_string(k) +
											  ",DOC GPU unified, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));
   
						delete dr;
						Experiment::testDone("DOC GPUnified Number of points: " + std::to_string(j*10) + " Dims used: " + std::to_string(k) + "Dim " + std::to_string(i));
					}else{
						Experiment::testDone("timeout");
					}
				}catch (std::exception& e){
					this->repportError("DOC GPU unified Exception caught : " + std::string(e.what()), this->getName());
					checkCudaErrors(cudaDeviceReset());								
				}
				
				/****************************** FAST DOC TESTS *****************************/

				try{
					if(true){
						DataReader* dr = new DataReader("testData/test1");
						Fast_DOCGPUnified c = Fast_DOCGPUnified(dr);
						c.setWidth(15);
						c.setSeed(seed);
						c.setAlpha(0.1);
						c.setFindDimVersion(chunksFindDim);
						c.setContainedVersion(LessReadingBreakContained);
						c.setBeta(0.25);

						// start timer
						auto t1 = std::chrono::high_resolution_clock::now();
						auto result = c.findKClusters(10);
						auto t2 = std::chrono::high_resolution_clock::now();
						// stop timer
						auto time = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
						float acc = 0;


						
						this->writeLineToFile(std::to_string(10) + ", "
											  + std::to_string(j*10) + ", "
											  +  std::to_string(i) + ", "
											  + std::to_string(k) +
											  ",FastDOC GPU unified, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));

						delete dr;
						Experiment::testDone("FastDOC GPUnified Number of points: " + std::to_string(j*10) + " Dims used: " + std::to_string(k) + "Dim " + std::to_string(i));
					}else{
						Experiment::testDone("timeout");
					}
				}catch (std::exception& e){
					this->repportError("Fast_DOC GPU Exception caught : " + std::string(e.what()), this->getName());
					checkCudaErrors(cudaDeviceReset());								
				}


				
				
				/****************************** MINECLUS TESTS *****************************/
				


				try{
					if(!MineClusGPUnifiedStop && i < 128){
						DataReader* dr = new DataReader("testData/test1");
						MineClusGPUnified c = MineClusGPUnified(dr);
						c.setWidth(15);
						c.setSeed(seed);
						c.setAlpha(0.1);
						c.setBeta(0.25);
						c.setConcurentVersion(false);
						c.setDuplicatesVersion(Hash);
						c.setCountSupportVersion(SmemCount);

						// start timer
						auto t1 = std::chrono::high_resolution_clock::now();
						auto result = c.findKClusters(10);
						auto t2 = std::chrono::high_resolution_clock::now();
						// stop timer
						auto time = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
						float acc = 0;


						this->writeLineToFile(std::to_string(10) + ", "
											  + std::to_string(j*10) + ", "
											  +  std::to_string(i) + ", "
											  + std::to_string(k) +
											  ",MineClus GPU unified non-concurrent, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));

						delete dr;
						Experiment::testDone("MineClus GPUnified non-concurent Number of points: " + std::to_string(j*10) + " Dims used: " + std::to_string(k) + "Dim " + std::to_string(i));
					}else{
						Experiment::testDone("timeout");
					}
				}catch (std::exception& e){
					this->repportError("MineClus GPU unified Exception caught : " + std::string(e.what()), this->getName());
					checkCudaErrors(cudaDeviceReset());								
				}

				try{
					if(true){
						DataReader* dr = new DataReader("testData/test1");
						MineClusGPUnified c = MineClusGPUnified(dr);
						c.setWidth(15);
						c.setSeed(seed);
						c.setAlpha(0.1);
						c.setBeta(0.25);
						c.setConcurentVersion(false);
						c.setDuplicatesVersion(Hash);
						c.setCountSupportVersion(SmemCount);

						// start timer
						auto t1 = std::chrono::high_resolution_clock::now();
						auto result = c.findKClusters(10);
						auto t2 = std::chrono::high_resolution_clock::now();
						// stop timer
						auto time = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
						float acc = 0;
						this->writeLineToFile(std::to_string(10) + ", "
											  + std::to_string(j*10) + ", "
											  +  std::to_string(i) + ", "
											  + std::to_string(k) +
											  ",MineClus GPU unified concurrent, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));
						delete dr;
						Experiment::testDone("MineClus GPUnified concurent Number of points: " + std::to_string(j*10) + " Dims used: " + std::to_string(k) + "Dim " + std::to_string(i));
					}else{
						Experiment::testDone("timeout");
					}
				}catch (std::exception& e){
					this->repportError("MineClus GPU unified concurent Exception caught : " + std::string(e.what()), this->getName());
					checkCudaErrors(cudaDeviceReset());								
				}


				checkCudaErrors(cudaDeviceReset());
			}
		}
	}
	
	

	//this->repportError("Error stub (Fake error)", this->getName());
	


	

	
	Experiment::stop();
};




