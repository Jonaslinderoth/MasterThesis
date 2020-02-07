/*
 * MetaDataFileReader.cpp
 *
 *  Created on: Feb 6, 2020
 *      Author: mikkel
 */

#include <src/testingTools/MetaDataFileReader.h>

#include <fstream>
#include <iostream>


MetaDataFileReader::MetaDataFileReader(std::string fileName_) {
	fileName = fileName_;
	howLongOnTheCheatVector = 0;
	numberOfClusters = 0;
	dimensions = 0;
	std::ifstream infile;
	std::string metaDataFileName = "meta_data_" + fileName + ".txt";
	infile.open(metaDataFileName);
	if (infile.is_open()) {
		std::string line;
		unsigned int lineNumber = 1;
		while(getline (infile,line)){
			if(lineNumber == 1){
				dimensions = std::stoi(line);
			}else if(lineNumber == 2){
				numberOfClusters = std::stoi(line);
			}else if(lineNumber > 2 and lineNumber <= (2+numberOfClusters) ){
				clusterLines.push_back(line);
				Cluster cluster;
				std::vector<std::string> parts = this->splitString(line,',');
				for(std::vector<std::string>::iterator iter = parts.begin() ; iter != parts.end() ; ++iter){
					if((*iter).compare("u") == 0){
						BoundsForUniformDistribution boundsForUniformDistribution;
						float constant;
						++iter;
						boundsForUniformDistribution.lower = std::stof(*iter);
						++iter;
						boundsForUniformDistribution.upper = std::stof(*iter);
						++iter;
						cluster.addDimension(uniformDistribution,boundsForUniformDistribution);
					}else if((*iter).compare("n") == 0){
						BoundsForUniformDistribution boundsForUniformDistribution;
						MeanAndVarianceForNormalDistribution meanAndVarianceForNormalDistribution;
						++iter;
						meanAndVarianceForNormalDistribution.mean = std::stof(*iter);
						++iter;
						meanAndVarianceForNormalDistribution.variance = std::stof(*iter);
						cluster.addDimension(normalDistribution,boundsForUniformDistribution,meanAndVarianceForNormalDistribution);
					}else if((*iter).compare("c") == 0){
						BoundsForUniformDistribution boundsForUniformDistribution;
						MeanAndVarianceForNormalDistribution meanAndVarianceForNormalDistribution;
						float constant;
						++iter;
						constant = std::stof(*iter);
						cluster.addDimension(normalDistribution,boundsForUniformDistribution,meanAndVarianceForNormalDistribution,constant);
					}else{
						cluster.setAmmount(std::stoi(*iter));
						clusters.push_back(cluster);
					}
				}
			}else{
				cheatVector.push_back(std::stoi(line));
			}


			lineNumber++;
		}
		infile.close();
	}

}

std::vector<std::string> MetaDataFileReader::splitString(const std::string str, char delim )
{
	std::vector<std::string> res;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delim)) {
        res.push_back(token);
    }
    return res;
}

unsigned int MetaDataFileReader::nextCheat(){
	unsigned int res = cheatVector.at(howLongOnTheCheatVector);
	howLongOnTheCheatVector++;
	return res;
}


unsigned int MetaDataFileReader::getDimensions() {
	return dimensions;
}

MetaDataFileReader::~MetaDataFileReader() {
	// TODO Auto-generated destructor stub
}

std::vector<Cluster> MetaDataFileReader::getClusters() {
	return clusters;
}

std::vector<std::string> MetaDataFileReader::getClusterLines() {
	return clusterLines;
}
