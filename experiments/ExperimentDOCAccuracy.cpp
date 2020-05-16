#include "ExperimentDOCAccuracy.h"
#include <unistd.h>
#include <iostream>
#include "../src/testingTools/DataGeneratorBuilder.h"
#include "../src/dataReader/DataReader.h"
#include "../src/DOC_GPU/DOCGPU.h"
#include "../src/Evaluation.h"
#include "../src/randomCudaScripts/Utils.h"
#include <chrono>
#include "random"

void ExperimentDOCAccuracy::start(){
	unsigned int dim = 20;
	unsigned int numberOfPointsPerCluster = 128;
	unsigned int usedDim = 4;
	unsigned int c = 0;
	unsigned int numberOfSamples = 1048576;
	// Count number of tests
	for(unsigned int i = 1024; i <= numberOfSamples; i *= 2){
		c++;;
	}

	Experiment::addTests(c);
	
	Experiment::start();
	int seed = rand();
	// create dataBuilder
	DataGeneratorBuilder dgb;
	dgb.setSeed(seed);
	if(system("mkdir testData  >>/dev/null 2>>/dev/null")){
					
	};
	bool res = dgb.buildUClusters("testData/test1",numberOfPointsPerCluster,10,15,dim,usedDim,5, true);
	auto labels = Evaluation::getCluster("testData/test1");
	for(unsigned int i = 1024; i <= numberOfSamples; i *= 2){
	

		try{
			DataReader* dr = new DataReader("testData/test1");
			DOCGPU c = DOCGPU(dr);
			c.setWidth(15);
			c.setSeed(seed);
			c.setAlpha(0.1);
			c.setBeta(0.25);
			c.setNumberOfSamples(i);

			// start timer
			auto t1 = std::chrono::high_resolution_clock::now();
			auto result = c.findKClusters(10);
			auto t2 = std::chrono::high_resolution_clock::now();
			// stop timer
			auto time = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
			auto confusion = Evaluation::confusion(labels, result);
			auto acc = Evaluation::accuracy(confusion);
			this->writeLineToFile(std::to_string(10) + ", "
								  + std::to_string(numberOfPointsPerCluster*10) + ", "
								  +  std::to_string(dim) + ", "
								  + std::to_string(usedDim) + ", "
								  + std::to_string(i) + ", "
								  + std::to_string(time) + ", "
								  + std::to_string(result.size()) + ", "
								  + std::to_string(acc));
			delete dr;
			Experiment::testDone("DOC GPU sample size: " + std::to_string(i));
			if(time > 60000000*5){
				break;	
			}
		}catch (std::exception& e){
			this->repportError("DOC GPU Exception caught : " + std::string(e.what()), this->getName());
			checkCudaErrors(cudaDeviceReset());								
		}
	}
	checkCudaErrors(cudaDeviceReset());
	for(unsigned int g = 0; g < labels->size(); g++){
		delete labels->at(g);
	}
	delete labels;


	
	

	//this->repportError("Error stub (Fake error)", this->getName());
	


	

	
	Experiment::stop();
};




