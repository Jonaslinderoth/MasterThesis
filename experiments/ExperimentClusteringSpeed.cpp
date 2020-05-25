#include "ExperimentClusteringSpeed.h"
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

void ExperimentClusteringSpeed::start(){
	unsigned int dim = 128;
	unsigned int numberOfPointsPerCluster = 16384/2;
	unsigned int usedDim = 20;
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
	bool DOCCPUStop = false;
	bool DOCGPUStop = false;
	bool DOCGPUnifiedStop = false;
	bool FastDOCCPUStop = false;
	bool FastDOCGPUStop = false;
	bool FastDOCGPUnifiedStop = false;
	bool MineClusCPUStop = false;
	bool MineClusGPUStop = false;
	bool MineClusGPUnifiedStop = false;
	bool MineClusCPUCStop = false;
	bool MineClusGPUCStop = false;
	bool MineClusGPUnifiedCStop = false;
	int seed = 0;
	unsigned int timeout = 60000000/6;
	for(unsigned int i = 16; i <= dim; i *= 2){
		DOCCPUStop = false;
		DOCGPUStop = false;
		DOCGPUnifiedStop = false;
		FastDOCCPUStop = false;
		FastDOCGPUStop = false;
		FastDOCGPUnifiedStop = false;
		MineClusCPUStop = false;
		MineClusGPUStop = false;
		MineClusGPUnifiedStop = false;
		MineClusCPUCStop = false;
		MineClusGPUCStop = false;
		MineClusGPUnifiedCStop = false;
		for(unsigned int j = 32; j <= numberOfPointsPerCluster; j *=2){
			for(unsigned int k = 5; k <= usedDim; k +=10){
				if (k*2 > i) break;
				// create dataBuilder
				DataGeneratorBuilder dgb;
				dgb.setSeed(0);
				if(system("mkdir testData  >>/dev/null 2>>/dev/null")){
					
				};
				bool res = dgb.buildUClusters("testData/test1",j,10,15,i,k,5, true);
				std::vector<std::vector<std::vector<float>*>*>* labels;
				if(j <= 2048){
					labels = Evaluation::getCluster("testData/test1");
				}

				if(!res){this->repportError("Cluster not generated", this->getName());
				}else{
					Experiment::testDone("Cluster generated. Number of points: " + std::to_string(j*10) + " Dims used: " + std::to_string(k) + "Dim " + std::to_string(i));
				}
				seed = i*k*j;
				/******************************** DOC TESTS ********************************/
				try{
					if(!DOCCPUStop){
						DataReader* dr = new DataReader("testData/test1");
						DOC c = DOC(dr);
						c.setWidth(15);
						c.setSeed(seed);
						c.setAlpha(0.1);
						c.setBeta(0.25);
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
											  ",DOC CPU, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));
						if(time > timeout*2){
							DOCCPUStop = true;
						}
						delete dr;

						Experiment::testDone("DOC CPU Number of points: " + std::to_string(j*10) + " Dims used: " + std::to_string(k) + "Dim " + std::to_string(i));
					}else{
						Experiment::testDone("timeout");
					}
				}catch (std::exception& e){
					this->repportError("DOC Exception caught : " + std::string(e.what()), this->getName());
				}

				try{
					if(!DOCGPUStop){
						DataReader* dr = new DataReader("testData/test1");
						DOCGPU c = DOCGPU(dr);
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
											  ",DOC GPU, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));
						if(time > timeout){
							DOCGPUStop = true;
						}
						delete dr;
						Experiment::testDone("DOC GPU Number of points: " + std::to_string(j*10) + " Dims used: " + std::to_string(k) + "Dim " + std::to_string(i));
					}else{
						Experiment::testDone("timeout");
					}
				}catch (std::exception& e){
					this->repportError("DOC GPU Exception caught : " + std::string(e.what()), this->getName());
					checkCudaErrors(cudaDeviceReset());								
				}

				try{
					if(!DOCGPUnifiedStop){
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
						if(time > timeout){
							DOCGPUnifiedStop = true;
						}
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
					if(!FastDOCCPUStop){
						DataReader* dr = new DataReader("testData/test1");
						Fast_DOC c = Fast_DOC(dr);
						c.setWidth(15);
						c.setSeed(seed);
						c.setAlpha(0.1);
						c.setBeta(0.25);

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
											  ",FastDOC CPU, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));
						if(time > timeout){
							FastDOCCPUStop = true;
						}
						delete dr;
						Experiment::testDone("FastDOC CPU Number of points: " + std::to_string(j*10) + " Dims used: " + std::to_string(k) + "Dim " + std::to_string(i));
					}else{
						Experiment::testDone("timeout");
					}
				}catch (std::exception& e){
					this->repportError("Fast_DOC Exception caught : " + std::string(e.what()), this->getName());
				}

				try{
					if(!FastDOCGPUStop){
						DataReader* dr = new DataReader("testData/test1");
						Fast_DOCGPU c = Fast_DOCGPU(dr);
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
						if(j <= 2048){
							auto confusion = Evaluation::confusion(labels, result);
							acc = Evaluation::accuracy(confusion);
						}

						this->writeLineToFile(std::to_string(10) + ", "
											  + std::to_string(j*10) + ", "
											  +  std::to_string(i) + ", "
											  + std::to_string(k) +
											  ",FastDOC GPU, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));
						if(time > timeout){
							FastDOCGPUStop = true;
						}
						delete dr;
						Experiment::testDone("FastDOC GPU Number of points: " + std::to_string(j*10) + " Dims used: " + std::to_string(k) + "Dim " + std::to_string(i));
					}else{
						Experiment::testDone("timeout");
					}
				}catch (std::exception& e){
					this->repportError("Fast_DOC GPU Exception caught "+std::to_string(i)+ " "+std::to_string(j)+" "+std::to_string(k)+": " + std::string(e.what()), this->getName());
					checkCudaErrors(cudaDeviceReset());								
				}


				try{
					if(!FastDOCGPUnifiedStop){
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
						if(j <= 2048){
						auto confusion = Evaluation::confusion(labels, result);
						acc = Evaluation::accuracy(confusion);
						}

						
						this->writeLineToFile(std::to_string(10) + ", "
											  + std::to_string(j*10) + ", "
											  +  std::to_string(i) + ", "
											  + std::to_string(k) +
											  ",FastDOC GPU unified, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));
						if(time > timeout){
							FastDOCGPUnifiedStop = true;
						}
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
					if(!MineClusCPUStop && i < 128){
						DataReader* dr = new DataReader("testData/test1");
						MineClus c = MineClus(dr);
						c.setWidth(15);
						c.setSeed(seed);
						c.setAlpha(0.1);
						c.setBeta(0.25);
						c.setConcurentVersion(false);

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
											  ",MineClus CPU non-concurrent, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));
						if(time > timeout){
							MineClusCPUStop = true;
						}
						delete dr;
						Experiment::testDone("MineClus CPU non-concurent Number of points: " + std::to_string(j*10) + " Dims used: " + std::to_string(k) + "Dim " + std::to_string(i));
					}else{
						Experiment::testDone("timeout");
					}
				}catch (std::exception& e){
					this->repportError("MineClus Exception caught : " + std::string(e.what()), this->getName());						
				}

				try{
					if(!MineClusCPUCStop && i < 128){
						DataReader* dr = new DataReader("testData/test1");
						MineClus c = MineClus(dr);
						c.setWidth(15);
						c.setSeed(seed);
						c.setAlpha(0.1);
						c.setBeta(0.25);
						c.setConcurentVersion(false);

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
											  ",MineClus CPU concurrent, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));
						if(time > timeout){
							MineClusCPUCStop = true;
						}
						delete dr;
						Experiment::testDone("MineClus CPU concurent Number of points: " + std::to_string(j*10) + " Dims used: " + std::to_string(k) + "Dim " + std::to_string(i));
					}else{
						Experiment::testDone("timeout");
					}
				}catch (std::exception& e){
					this->repportError("MineClus concurent Exception caught : " + std::string(e.what()), this->getName());
				}

				try{
					if(!MineClusGPUStop && i < 128){
						DataReader* dr = new DataReader("testData/test1");
						MineClusGPU c = MineClusGPU(dr);
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
						if(j <= 2048){
							auto confusion = Evaluation::confusion(labels, result);
							acc = Evaluation::accuracy(confusion);
						}

						this->writeLineToFile(std::to_string(10) + ", "
											  + std::to_string(j*10) + ", "
											  +  std::to_string(i) + ", "
											  + std::to_string(k) +
											  ",MineClus GPU non-concurrent, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));
						if(time > timeout){
							MineClusGPUStop = true;
						}
						delete dr;
						Experiment::testDone("MineClus GPU non-concurent Number of points: " + std::to_string(j*10) + " Dims used: " + std::to_string(k) + "Dim " + std::to_string(i));
					}else{
						Experiment::testDone("timeout");
					}
				}catch (std::exception& e){
					this->repportError("MineClus GPU Exception caught : " + std::string(e.what()), this->getName());
					checkCudaErrors(cudaDeviceReset());								
				}
				
				try{
					if(!MineClusGPUCStop && i < 128){
						DataReader* dr = new DataReader("testData/test1");
						MineClusGPU c = MineClusGPU(dr);
						c.setWidth(15);
						c.setSeed(seed);
						c.setAlpha(0.1);
						c.setBeta(0.25);
						c.setConcurentVersion(true);
						c.setDuplicatesVersion(Hash);
						c.setCountSupportVersion(SmemCount);

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
											  ",MineClus GPU concurrent, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));
						if(time > timeout){
							MineClusGPUCStop = true;
						}
						delete dr;
						Experiment::testDone("MineClus GPU concurent Number of points: " + std::to_string(j*10) + " Dims used: " + std::to_string(k) + "Dim " + std::to_string(i));
						checkCudaErrors(cudaDeviceReset());								
					}else{
						Experiment::testDone("timeout");
					}
				}catch (std::exception& e){
					this->repportError("MineClus GPU concurent Exception caught : " + std::string(e.what()), this->getName());
				}

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
						if(j <= 2048){
						auto confusion = Evaluation::confusion(labels, result);
						acc = Evaluation::accuracy(confusion);
						}

						this->writeLineToFile(std::to_string(10) + ", "
											  + std::to_string(j*10) + ", "
											  +  std::to_string(i) + ", "
											  + std::to_string(k) +
											  ",MineClus GPU unified non-concurrent, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));
						if(time > timeout){
							MineClusGPUnifiedStop = true;
						}
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
					if(!MineClusGPUnifiedCStop && i < 128){
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
						if(j <= 2048){
							auto confusion = Evaluation::confusion(labels, result);
							acc = Evaluation::accuracy(confusion);
						}

						
						this->writeLineToFile(std::to_string(10) + ", "
											  + std::to_string(j*10) + ", "
											  +  std::to_string(i) + ", "
											  + std::to_string(k) +
											  ",MineClus GPU unified concurrent, "
											  + std::to_string(time) + ", "
											  + std::to_string(result.size()) + ", "
											  + std::to_string(acc));
						if(time > timeout){
							MineClusGPUnifiedCStop = true;
						}
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
				if(j <= 2048){
					for(unsigned int g = 0; g < labels->size(); g++){
						delete labels->at(g);
					}
					delete labels;
				}
			}
		}
	}
	
	

	//this->repportError("Error stub (Fake error)", this->getName());
	


	

	
	Experiment::stop();
};




