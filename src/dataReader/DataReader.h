/*
 * DataReader.h
 *
 *  Created on: Jan 29, 2020
 *      Author: mikkel
 */

#ifndef DATAREADER_H_
#define DATAREADER_H_

#include <string>
#include <mutex>
#include <vector>
#include <queue>
#include <iostream>
#include <thread>

#include <src/dataReader/DataReaderQueue.h>

class DataReader {
public:
	DataReader(std::string fileName_ = "test1", unsigned int queSize_ = 32, unsigned int chunkSize_ = 65536 );
	bool isNextReady();
	bool isThereANextBlock();
	std::vector<std::vector<float>*>* next();
	bool isThereANextPoint();
	std::vector<float>* nextPoint();
	void printQueue();
	virtual ~DataReader();
	void badRead();
	unsigned int getSize();
	unsigned int getDimensions();
private:
	unsigned int howLongOnTheNextBlock;
	std::vector<std::vector<float>*>* currentBlock;
	std::thread t;
	DataReaderQueue* dataReaderQueuePointer;
	std::queue<std::vector<std::vector<float>*>*> queuePointer;
	unsigned int chunkSize;
	unsigned int queSize;
	std::string fileName;
};

#endif /* DATAREADER_H_ */
