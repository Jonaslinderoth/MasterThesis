/*
 * DataReaderQueue.h
 *
 *  Created on: Feb 4, 2020
 *      Author: mikkel
 */

#ifndef DATAREADERQUEUE_H_
#define DATAREADERQUEUE_H_

#include <string>
#include <queue>
#include <vector>
#include <mutex>

class DataReaderQueue {
public:
	DataReaderQueue(std::string fileName_ , unsigned int queSize_ = 8, unsigned int chunkSize_ = 1024);
	void run();
	bool isThereMoreData();
	virtual ~DataReaderQueue();
	std::vector<std::vector<float>*>* popFromQue();
	unsigned int getSize();
	unsigned int getDimensions();
	unsigned int getQueueSize();
private:
	bool theThreadIsDone;
	std::mutex mtx;
	bool pushToQue(std::vector<std::vector<float>*>* block);
	std::queue<std::vector<std::vector<float>*>*> myQueue;
	unsigned int chunkSize;
	unsigned int queSize;
	unsigned int size;
	unsigned int numberOfDimensions;
	FILE* file;
	std::string fileName;
};


#endif /* DATAREADERQUEUE_H_ */
