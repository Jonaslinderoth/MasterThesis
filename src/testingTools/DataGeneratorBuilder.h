/*
 * DataGeneratorBuilder.h
 *
 *  Created on: Jan 29, 2020
 *      Author: mikkel
 */

#ifndef DATAGENERATORBUILDER_H_
#define DATAGENERATORBUILDER_H_

#include <src/dataReader/Cluster.h>
#include <src/testingTools/DataGenerator.h>
#include <vector>

class DataGeneratorBuilder {
public:
	DataGeneratorBuilder();
	/*
	 * if no dimension is chosen the it take the one from the biggest cluster
	 * you can not set a dimension that is smaller than one of the current clusters
	 */
	bool setDimension(unsigned int dimension_);
	bool addCluster(Cluster cluster = Cluster());
	bool setFileName(std::string fileName_);
	bool build();
	bool deleteFiles(std::vector<std::string> vecOfFilesNames);
	virtual ~DataGeneratorBuilder();
	bool buildUClusters(unsigned int ammountOfPoint = 100000 , unsigned int ammountOfClusters = 5 , unsigned int with = 15 , unsigned int dimensions = 200 , unsigned int dimensionUsed = 20 , float outLiersPersentage = 5);
	bool buildMGqClusters(unsigned int q = 1 , unsigned int ammountOfPoint = 100000 , unsigned int ammountOfClusters = 5 , unsigned int with = 15 , unsigned int dimensions = 200 , unsigned int dimensionUsed = 20 , float outLiersPersentage = 5);
private:
	bool spitFiles(std::string fileName ,
			const unsigned int numberOfDimensions ,
			const unsigned int numberOfClusters ,
			std::vector<std::vector<DistributionType>> distribuitionTypeForEachClusterForEachDimension,
			std::vector<std::vector<BoundsForUniformDistribution>> uniBoundsForEachClusterForEachDimension,
			std::vector<std::vector<MeanAndVarianceForNormalDistribution>> meanAndVarianceForNormalDistributionForEachClusterForEachDimension,
			std::vector<std::vector<float>> constantForEachClusterForEachDimension,
			std::vector<unsigned int> numberOfPointForEachCluster);
	std::string fileName;
	unsigned int dimension;
	bool dimensionSet;
	std::vector<Cluster> allClusters;

};

#endif /* DATAGENERATORBUILDER_H_ */
