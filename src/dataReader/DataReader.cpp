/*
 * DataReader.cpp
 *
 *  Created on: Jan 29, 2020
 *      Author: mikkel
 */

#include <src/dataReader/DataReader.h>

#include <fstream>
#include <stdio.h>
#include <random>
#include <cstring>

#include <src/dataReader/DataReaderQueue.h>


DataReader::DataReader(std::string fileName_ , unsigned int queSize_, unsigned int chunkSize_) {
	chunkSize = chunkSize_;
	queSize = queSize_;
	fileName = fileName_;
	howLongOnTheNextBlock = 0;
	currentBlock = new std::vector<std::vector<float>*>;
	dataReaderQueuePointer = new DataReaderQueue(fileName,queSize,chunkSize);
	t = std::thread(&DataReaderQueue::run,dataReaderQueuePointer);

}

std::vector<std::vector<float>*>* DataReader::next(){
	return dataReaderQueuePointer->popFromQue();
}

bool DataReader::isNextReady(){
	if(dataReaderQueuePointer->getQueueSize()>0){
		return true;
	}
	return false;
}

void DataReader::printQueue() {
	float sum = 0;
	while(dataReaderQueuePointer->isThereMoreData()){
		std::vector<std::vector<float>*>* dataBlock;
		dataBlock = dataReaderQueuePointer->popFromQue();

		for(std::vector<std::vector<float>*>::iterator iter = dataBlock->begin() ; iter != dataBlock->end() ; ++iter)
		{
			for(std::vector<float>::iterator innerIter = (*iter)->begin() ; innerIter != (*iter)->end() ; ++innerIter)
			{
				sum += *innerIter;

			}

		}
	}
	std::cout << "sum " << std::to_string(sum) << std::endl;
}

unsigned int DataReader::getSize(){
	return dataReaderQueuePointer->getSize();
}

unsigned int DataReader::getDimensions(){
	return dataReaderQueuePointer->getDimensions();
}
bool DataReader::isThereANextBlock(){
	return dataReaderQueuePointer->isThereMoreData();
}

bool DataReader::isThereANextPoint(){
	if(currentBlock->size() > howLongOnTheNextBlock){
		return true;
	}
	return dataReaderQueuePointer->isThereMoreData();
}

std::vector<float>* DataReader::nextPoint(){
	if(howLongOnTheNextBlock >= currentBlock->size()){
		delete currentBlock;
		if(this->isThereANextBlock()){
			currentBlock = this->next();
			howLongOnTheNextBlock = 0;
		}else{
			std::cout << "you are trying to get a nextPoint without checking with isThereANextPoint!" << std::endl;
			return NULL;
		}
	}
	std::vector<float>* res = currentBlock->at(howLongOnTheNextBlock);
	howLongOnTheNextBlock++;
	return res;

}







DataReader::~DataReader() {

	t.join();
	delete dataReaderQueuePointer;
}


/*
 * this shoud not be used , it probably memory leaks and stuff
 */
void DataReader::badRead(){
	std::string binaryFileName = fileName + ".dat";
	char cstrFileName[binaryFileName.size() + 1];
	std::strcpy(cstrFileName, binaryFileName.c_str());

	FILE* file = fopen(cstrFileName, "rb");


	float fnumberOfDimensions;
	fread(&fnumberOfDimensions, sizeof(float), 1, file);
	float fsize;
	fread(&fsize, sizeof(float), 1, file);
	unsigned int size = (unsigned int)fsize;
	unsigned int numberOfDimensions = (unsigned int)fnumberOfDimensions;


	std::vector<std::vector<float>> data;

	for(int dataIndex = 0 ; dataIndex < size ; ++dataIndex)
	{
		std::vector<float> dataPoint;
		for(int dimensionIndex = 0 ; dimensionIndex < numberOfDimensions ; ++dimensionIndex)
		{
			float f;
			fread(&f, sizeof(float), 1, file);
			dataPoint.push_back(f);

		}
		data.push_back(dataPoint);
	}

	fclose(file);

	float sum = 0;
	for(std::vector<std::vector<float>>::iterator iter = data.begin() ; iter != data.end() ; ++iter)
	{
		for(std::vector<float>::iterator innerIter = iter->begin() ; innerIter != iter->end() ; ++innerIter)
		{
			sum += *innerIter;
		}
	}
	std::cout << "sum " << std::to_string(sum) << std::endl;
}



