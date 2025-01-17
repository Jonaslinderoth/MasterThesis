/*
 * DataGenerator.cpp
 *
 *  Created on: Jan 25, 2020
 *      Author: mikkel
 */

#include <src/testingTools/DataGenerator.h>
#include <src/testingTools/RandomFunction.h>

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <cstring>
#include <algorithm>
#include <ctime>
#include <new>
#include <math.h>


struct DataUnitStruct{
	float dataUnit;
	int clusterIndex;
};

DataGenerator::DataGenerator(std::string fileName ,
		const unsigned int numberOfDimensions ,
		const unsigned int numberOfClusters ,
		std::vector<std::vector<DistributionType>> distribuitionTypeForEachClusterForEachDimension,
		std::vector<std::vector<BoundsForUniformDistribution>> uniBoundsForEachClusterForEachDimension,
		std::vector<std::vector<MeanAndVarianceForNormalDistribution>> meanAndVarianceForNormalDistributionForEachClusterForEachDimension,
		std::vector<std::vector<float>> constantForEachClusterForEachDimension,
		std::vector<unsigned int> numberOfPointForEachCluster,
		unsigned int metaDataStartClusterIndex,
		PreviusClustersInformation* previusClustersInformation_){

	previusClustersInformation = previusClustersInformation_;
	std::vector<std::vector<DataUnitStruct>> data;

	//this is to store things for the next iteration
	std::vector<float> vectorOfPreviusCentroids;
	std::vector<float> vectorOfPreviusVariance;
	std::vector<bool> vectorUsedNormalDistribuitionBefore;

	for(int dimensionIndex = 0 ; dimensionIndex < numberOfDimensions ; dimensionIndex++){
		vectorOfPreviusCentroids.push_back(meanAndVarianceForNormalDistributionForEachClusterForEachDimension.at(0).at(dimensionIndex).mean);
		vectorOfPreviusVariance.push_back(meanAndVarianceForNormalDistributionForEachClusterForEachDimension.at(0).at(dimensionIndex).variance);
		vectorUsedNormalDistribuitionBefore.push_back(true);
	}


	for(int clusterIndex = 0 ; clusterIndex < numberOfClusters ; clusterIndex++){
		//how many points we are going to generate for this cluster
		int numberOfPointInCluster = numberOfPointForEachCluster.at(clusterIndex);

		//if we have prior information.
		if(previusClustersInformation_ != nullptr){
			vectorOfPreviusCentroids = previusClustersInformation->vectorOfPreviusCentroids;
			vectorOfPreviusVariance = previusClustersInformation->vectorOfPreviusVariance;
			vectorUsedNormalDistribuitionBefore = previusClustersInformation->vectorUsedNormalDistribuitionBefore;

		}

		for(int numberOfPointInClusterIndex = 0 ; numberOfPointInClusterIndex < numberOfPointInCluster ; numberOfPointInClusterIndex++){
			//this is a point going to have numberOfDimensions dataUnits
			std::vector<DataUnitStruct> dataPoint;


			//for each dimension we need to make a dataUnit
			for(int dimensionIndex = 0 ; dimensionIndex < numberOfDimensions ; dimensionIndex++){
				//this is what we are trying to generate
				DataUnitStruct dataUnitStruct;
				//check what type of distribuition we need to use
				DistributionType distributionTypeForDataUnit = distribuitionTypeForEachClusterForEachDimension.at(clusterIndex).at(dimensionIndex);
				if(distributionTypeForDataUnit == uniformDistribution)
				{
					BoundsForUniformDistribution boundsForUniformDistributionForDataUnit = uniBoundsForEachClusterForEachDimension.at(clusterIndex).at(dimensionIndex);
					dataUnitStruct.dataUnit = RandomFunction::uniformRandomFloat(boundsForUniformDistributionForDataUnit.lower,boundsForUniformDistributionForDataUnit.upper);
					dataUnitStruct.clusterIndex = clusterIndex;

				}else if(distributionTypeForDataUnit == normalDistribution){
					MeanAndVarianceForNormalDistribution meanAndVarianceForNormalDistribution = meanAndVarianceForNormalDistributionForEachClusterForEachDimension.at(clusterIndex).at(dimensionIndex);
					float mean = meanAndVarianceForNormalDistribution.mean;
					float var = meanAndVarianceForNormalDistribution.variance;
					dataUnitStruct.dataUnit =  RandomFunction::normalDistributionRandomFloat(mean,var);
					dataUnitStruct.clusterIndex = clusterIndex;

				}else if(distributionTypeForDataUnit == normalDistributionSpecial){
					MeanAndVarianceForNormalDistribution meanAndVarianceForNormalDistribution = meanAndVarianceForNormalDistributionForEachClusterForEachDimension.at(clusterIndex).at(dimensionIndex);
					BoundsForUniformDistribution boundsForUniformDistributionForDataUnit = uniBoundsForEachClusterForEachDimension.at(clusterIndex).at(dimensionIndex);
					float center = meanAndVarianceForNormalDistribution.variance;
					float var = meanAndVarianceForNormalDistribution.variance;
					float lower = boundsForUniformDistributionForDataUnit.lower;
					float upper = boundsForUniformDistributionForDataUnit.upper;
					unsigned int q = meanAndVarianceForNormalDistribution.q;
					float specialVar = (sqrt(var)/(float)q)*(sqrt(var)/(float)q);
					if(vectorUsedNormalDistribuitionBefore.at(dimensionIndex)){
						float prevCenter = vectorOfPreviusCentroids.at(dimensionIndex);
						float prevVar = vectorOfPreviusVariance.at(dimensionIndex);
						float stdDev = sqrt(prevVar);
						specialVar = (stdDev/(float)q)*(stdDev/(float)q);
						float upperMax = boundsForUniformDistributionForDataUnit.upper;
						upper = prevCenter+2*stdDev;
						if(upper > upperMax){
							upper = upperMax;
						}
						center = std::fmod(RandomFunction::randomInteger(),upper-lower)+lower;
					}
					if(meanAndVarianceForNormalDistribution.q == 1){
						dataUnitStruct.dataUnit =  RandomFunction::normalDistributionRandomFloat(center,var);
					}else{
						unsigned int whatCenteToGenerateTo = RandomFunction::randomInteger()%q;
						float with = upper-lower;
						center = (whatCenteToGenerateTo+1)*(with/(q+1))+lower;
						dataUnitStruct.dataUnit = RandomFunction::normalDistributionRandomFloat(center,specialVar);
					}

					dataUnitStruct.clusterIndex = clusterIndex;

				}else if(distributionTypeForDataUnit == constant)
				{
					dataUnitStruct.dataUnit = constantForEachClusterForEachDimension.at(clusterIndex).at(dimensionIndex);
					dataUnitStruct.clusterIndex = clusterIndex;
				}

				//TODO add other distribuitions

				dataPoint.push_back(dataUnitStruct);

			}


			//now that we have a point we add that point to the data
			data.push_back(dataPoint);
		}
		//need to set if the next cluster needs to use the mean and variance.
		for(int dimensionIndex = 0 ; dimensionIndex < numberOfDimensions ; dimensionIndex++){
			DistributionType distributionTypeForDataUnit = distribuitionTypeForEachClusterForEachDimension.at(clusterIndex).at(dimensionIndex);

			if(distributionTypeForDataUnit == uniformDistribution){
				vectorUsedNormalDistribuitionBefore.at(dimensionIndex) = false;

			}else if(distributionTypeForDataUnit == normalDistribution){
				vectorUsedNormalDistribuitionBefore.at(dimensionIndex) = false;

			}else if(distributionTypeForDataUnit == normalDistributionSpecial){
				vectorUsedNormalDistribuitionBefore.at(dimensionIndex) = true;

			}else if(distributionTypeForDataUnit == constant){
				vectorUsedNormalDistribuitionBefore.at(dimensionIndex) = false;

			}
		}


		//need to calculate mean and variance
		for(int dimensionIndex = 0 ; dimensionIndex < numberOfDimensions ; dimensionIndex++){
			vectorOfPreviusCentroids.at(dimensionIndex) = 0;
			vectorOfPreviusVariance.at(dimensionIndex) = 0;
		}
		for(int numberOfPointInClusterIndex = 0 ; numberOfPointInClusterIndex < numberOfPointForEachCluster.at(clusterIndex) ; numberOfPointInClusterIndex++){
			for(int dimensionIndex = 0 ; dimensionIndex < numberOfDimensions ; dimensionIndex++){

				vectorOfPreviusCentroids.at(dimensionIndex) += data.at(numberOfPointInClusterIndex).at(dimensionIndex).dataUnit;
			}
		}
		for(int dimensionIndex = 0 ; dimensionIndex < numberOfDimensions ; dimensionIndex++){
			vectorOfPreviusCentroids.at(dimensionIndex) = vectorOfPreviusCentroids.at(dimensionIndex)/numberOfPointForEachCluster.at(clusterIndex);
		}
		for(int numberOfPointInClusterIndex = 0 ; numberOfPointInClusterIndex < numberOfPointForEachCluster.at(clusterIndex) ; numberOfPointInClusterIndex++){
			for(int dimensionIndex = 0 ; dimensionIndex < numberOfDimensions ; dimensionIndex++){
				float mean = vectorOfPreviusCentroids.at(dimensionIndex);
				float point = data.at(numberOfPointInClusterIndex).at(dimensionIndex).dataUnit;
				vectorOfPreviusVariance.at(dimensionIndex) += (point-mean)*(point-mean);

			}
		}
		for(int dimensionIndex = 0 ; dimensionIndex < numberOfDimensions ; dimensionIndex++){
			vectorOfPreviusVariance.at(dimensionIndex) = vectorOfPreviusVariance.at(dimensionIndex)/(numberOfPointForEachCluster.at(clusterIndex)-1);
			float var = vectorOfPreviusVariance.at(dimensionIndex);
			//std::cout << "var " << var << std::endl;

		}
	}

	if(previusClustersInformation != nullptr){
		delete previusClustersInformation;
	}
	previusClustersInformation = new PreviusClustersInformation;
	previusClustersInformation->vectorOfPreviusCentroids = vectorOfPreviusCentroids;
	previusClustersInformation->vectorOfPreviusVariance = vectorOfPreviusVariance;
	previusClustersInformation->vectorUsedNormalDistribuitionBefore = vectorUsedNormalDistribuitionBefore;

	//now we have to shuffle the data.
	std::srand ( RandomFunction::randomInteger() );
	std::random_shuffle ( data.begin(), data.end() );


	std::string binaryFileName = fileName + ".dat";
	//now we have the data , we need to write it to file.
	char cstrFileName[binaryFileName.size() + 1];
	std::strcpy(cstrFileName, binaryFileName.c_str());

	//start by deleting the previus file
	std::remove(cstrFileName);
	FILE* file = fopen (cstrFileName, "wb");
	if(file == nullptr){
		std::cout << "was not able to make: " << binaryFileName << std::endl;
		throw 1;
	}
	//metadata on the binary file
	float fnumberOfDimensions = (float)numberOfDimensions;
	fwrite(&fnumberOfDimensions, sizeof(float), 1, file);
	float fsize = (float)data.size();
	fwrite(&fsize, sizeof(float), 1, file);


	for(std::vector<std::vector<DataUnitStruct>>::iterator iter = data.begin(); iter != data.end() ; ++iter)
	{
		for(std::vector<DataUnitStruct>::iterator innerInter = iter->begin() ; innerInter != iter->end() ; ++innerInter)
		{
			float f = innerInter->dataUnit;
			fwrite(&f, sizeof(float), 1, file);
		}
	}

	fclose(file);


	//the meta in the text file
	std::string metaDataFileName = fileName+ "_meta_data" + ".txt";

	char cstrMetaDataFileName[metaDataFileName.size() + 1];
	std::strcpy(cstrMetaDataFileName, metaDataFileName.c_str());
	std::remove(cstrMetaDataFileName);
	std::ofstream outfile;
	outfile.open(metaDataFileName);


	outfile << std::to_string(numberOfDimensions) << std::endl;
	outfile << std::to_string(numberOfClusters) << std::endl;

	for(int clusterIndex = 0 ; clusterIndex < numberOfClusters ; clusterIndex++){
		for(int dimensionIndex = 0 ; dimensionIndex < numberOfDimensions ; dimensionIndex++){
			DistributionType distributionTypeForDataUnit = distribuitionTypeForEachClusterForEachDimension.at(clusterIndex).at(dimensionIndex);
			if(distributionTypeForDataUnit == uniformDistribution){
				outfile << "u,";
				BoundsForUniformDistribution boundsForUniformDistributionForDataUnit = uniBoundsForEachClusterForEachDimension.at(clusterIndex).at(dimensionIndex);
				outfile << std::to_string(boundsForUniformDistributionForDataUnit.lower) << "," << std::to_string(boundsForUniformDistributionForDataUnit.upper) << ",";
			}else if(distributionTypeForDataUnit == normalDistribution){
				outfile << "n,";
				MeanAndVarianceForNormalDistribution meanAndVarianceForNormalDistribution = meanAndVarianceForNormalDistributionForEachClusterForEachDimension.at(clusterIndex).at(dimensionIndex);
				outfile << std::to_string(meanAndVarianceForNormalDistribution.mean) << "," << std::to_string(meanAndVarianceForNormalDistribution.variance) << ",";
			}else if(distributionTypeForDataUnit == normalDistributionSpecial){
				outfile << "s,";
				MeanAndVarianceForNormalDistribution meanAndVarianceForNormalDistribution = meanAndVarianceForNormalDistributionForEachClusterForEachDimension.at(clusterIndex).at(dimensionIndex);
				outfile << std::to_string(meanAndVarianceForNormalDistribution.mean) << "," << std::to_string(meanAndVarianceForNormalDistribution.variance) << "," << std::to_string(meanAndVarianceForNormalDistribution.q) << ",";
			}else{
				outfile << "c,";
				outfile << std::to_string(constantForEachClusterForEachDimension.at(clusterIndex).at(dimensionIndex)) << ",";
			}

		}
		int numberOfPointInCluster = numberOfPointForEachCluster.at(clusterIndex);
		outfile << std::to_string(numberOfPointInCluster);
		outfile << std::endl;

	}


	for(std::vector<std::vector<DataUnitStruct>>::iterator iter = data.begin(); iter != data.end() ; ++iter)
	{
		outfile << std::to_string(iter->begin()->clusterIndex+metaDataStartClusterIndex) << std::endl;
	}
	outfile.close();
}



DataGenerator::~DataGenerator() {
	// TODO Auto-generated destructor stub
}

PreviusClustersInformation* DataGenerator::getPreviusClustersInformation() {
	return previusClustersInformation;
}
