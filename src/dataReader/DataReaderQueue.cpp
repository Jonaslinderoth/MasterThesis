/*
 * DataReaderQueue.cpp
 *
 *  Created on: Feb 4, 2020
 *      Author: mikkel
 */

#include <src/dataReader/DataReaderQueue.h>
#include <iostream>
#include <stdio.h>
#include <cstring>
#include <chrono>
#include <thread>


DataReaderQueue::DataReaderQueue(std::string fileName_ , unsigned int queSize_, unsigned int chunkSize_) {
	theThreadIsDone = false;
	chunkSize = chunkSize_ ;
	queSize = queSize_;
	fileName = fileName_;

	std::string binaryFileName = fileName + ".dat";
	char cstrFileName[binaryFileName.size() + 1];
	std::strcpy(cstrFileName, binaryFileName.c_str());

	file = fopen(cstrFileName, "rb");


	float fnumberOfDimensions;
	fread(&fnumberOfDimensions, sizeof(float), 1, file);
	float fsize;
	fread(&fsize, sizeof(float), 1, file);
	size = (unsigned int)fsize;
	numberOfDimensions = (unsigned int)fnumberOfDimensions;



}
void DataReaderQueue::run(){

	unsigned int dataPointsRead = 0;
	while(dataPointsRead < size){
		//this is the new block
		std::vector<std::vector<float>*>* dataBlock = new std::vector<std::vector<float>*>;
		//now adding to the block
		unsigned int dataPointsReadInBlock = 0;
		while(dataPointsReadInBlock < chunkSize and dataPointsRead < size){
			//this is a point that will contain d floats
			std::vector<float>* dataPoint = new std::vector<float>;
			for(int dimensionIndex = 0 ; dimensionIndex < numberOfDimensions ; ++dimensionIndex)
			{
				float f;
				fread(&f, sizeof(float), 1, file);
				dataPoint->push_back(f);
			}
			dataBlock->push_back(dataPoint);
			dataPointsRead++;
			dataPointsReadInBlock++;
		}
		//we need to try to push it until it get pushed to the que
		bool pushed = false;
		while(!pushed){
			//we dont want to fill the ram with blocks of data if we have many already
			if(this->getQueueSize() < queSize){
				//the data gets pushed to the que
				pushed = this->pushToQue(dataBlock);
			}else{
				/*
				 * if the queue is full we dont want to ask if it is full all the time
				 * pauseTime increases the more times we have asked , and increases the time between we ask.
				 */
				bool morePause = true;
				unsigned int pauseTime = 1;
				while(morePause){
					std::this_thread::sleep_for(std::chrono::milliseconds(pauseTime));
					if(this->getQueueSize() < queSize/2){
						morePause = false;
					}else{
						pauseTime++;
					}
				}
			}
		}


	}

	theThreadIsDone = true;
}

bool DataReaderQueue::pushToQue(std::vector<std::vector<float>*>* block){
	mtx.lock();
	myQueue.push(block);
	mtx.unlock();
	return true;
}

std::vector<std::vector<float>*>* DataReaderQueue::popFromQue(){
	bool thereIsSomething = false;
	unsigned int pauseTime = 0;
	while(!thereIsSomething){
		if(this->getQueueSize() > 0){
			thereIsSomething = true;
		}else{

			std::this_thread::sleep_for(std::chrono::milliseconds(pauseTime));
			pauseTime += 5;
			if(!this->isThereMoreData()){
				std::cout << "popFromQue while empty" << std::endl;
			}
		}
	}

	mtx.lock();
	std::vector<std::vector<float>*>* ret = myQueue.front();
	myQueue.pop();
	mtx.unlock();
	return ret;
}

unsigned int DataReaderQueue::getQueueSize(){
	mtx.lock();
	unsigned int ret = myQueue.size();
	mtx.unlock();
	return ret;
}

unsigned int DataReaderQueue::getDimensions(){
	return numberOfDimensions;
}

DataReaderQueue::~DataReaderQueue() {
	fclose(file);
}

bool DataReaderQueue::isThereMoreData() {
	if(theThreadIsDone == true and this->getQueueSize() == 0){
		return false;
	}
	return true;
}

unsigned int DataReaderQueue::getSize(){
	mtx.lock();
	int res =  size;
	mtx.unlock();
	return res;
}
