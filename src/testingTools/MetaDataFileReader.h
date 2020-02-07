/*
 * MetaDataFileReader.h
 *
 *  Created on: Feb 6, 2020
 *      Author: mikkel
 */

#ifndef METADATAFILEREADER_H_
#define METADATAFILEREADER_H_

#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iterator>

#include <src/dataReader/Cluster.h>





class MetaDataFileReader {
public:
	MetaDataFileReader(std::string fileName_ = "test1");
	unsigned int getDimensions();
	std::vector<Cluster> getClusters();
	virtual ~MetaDataFileReader();
	std::vector<std::string> splitString(const std::string str, char delim = ' ');
	unsigned int nextCheat();
	std::vector<std::string> getClusterLines();
private:
	unsigned int howLongOnTheCheatVector;
	std::vector<Cluster> clusters;
	std::vector<std::string> clusterLines;
	std::vector<unsigned int> cheatVector;
	std::string fileName;
	unsigned int dimensions;
	unsigned int numberOfClusters;
};

#endif /* METADATAFILEREADER_H_ */
