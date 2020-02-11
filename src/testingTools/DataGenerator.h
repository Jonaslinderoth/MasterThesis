/*
 * DataGenerator.h
 *
 *  Created on: Jan 25, 2020
 *      Author: mikkel
 */

#ifndef DATAGENERATOR_H_
#define DATAGENERATOR_H_

#include <src/dataReader/Cluster.h>
#include <string>
#include <vector>
#include <iostream>
#include <random>

static float uniformRandomFloat(float lowest , float max){
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<> distribution(lowest,max);
	float res = distribution(generator);
	return res;
}

static float normalDistributionRandomFloat(float mean , float variance){
	std::default_random_engine generator;
	generator.seed(rand());
	std::normal_distribution<float> distribution(mean,sqrt(variance));
	float res = distribution(generator);
	return res;
}

struct PreviusClustersInformation{
	std::vector<float> vectorOfPreviusCentroids;
	std::vector<float> vectorOfPreviusVariance;
	std::vector<bool> vectorUsedNormalDistribuitionBefore;
};


class DataGenerator {
public:
	DataGenerator(std::string fileName ,
			const unsigned int numberOfDimensions ,
			const unsigned int numberOfClusters ,
			std::vector<std::vector<DistributionType>> distribuitionTypeForEachClusterForEachDimension,
			std::vector<std::vector<BoundsForUniformDistribution>> uniBoundsForEachClusterForEachDimension,
			std::vector<std::vector<MeanAndVarianceForNormalDistribution>> meanAndVarianceForNormalDistributionForEachClusterForEachDimension,
			std::vector<std::vector<float>> constantForEachClusterForEachDimension,
			std::vector<unsigned int> numberOfPointForEachCluster,
			unsigned int metaDataStartClusterIndex = 0,
			PreviusClustersInformation* previusClustersInformation_ = nullptr);
	virtual ~DataGenerator();
	std::string getErrors();
	PreviusClustersInformation* getPreviusClustersInformation();
private:
	PreviusClustersInformation* previusClustersInformation;
};

#endif /* DATAGENERATOR_H_ */
