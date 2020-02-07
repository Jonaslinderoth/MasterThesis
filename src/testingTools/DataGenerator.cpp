/*
 * DataGenerator.cpp
 *
 *  Created on: Jan 25, 2020
 *      Author: mikkel
 */

#include <src/testingTools/DataGenerator.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <cstring>
#include <algorithm>
#include <ctime>
#include <new>


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
		unsigned int metaDataStartClusterIndex){


	std::vector<std::vector<DataUnitStruct>> data;

	for(int clusterIndex = 0 ; clusterIndex < numberOfClusters ; clusterIndex++){
		//how many points we are going to generate for this cluster
		int numberOfPointInCluster = numberOfPointForEachCluster.at(clusterIndex);

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
					dataUnitStruct.dataUnit = uniformRandomFloat(boundsForUniformDistributionForDataUnit.lower,boundsForUniformDistributionForDataUnit.upper);
					dataUnitStruct.clusterIndex = clusterIndex;
				}else if(distributionTypeForDataUnit == normalDistribution){
					MeanAndVarianceForNormalDistribution meanAndVarianceForNormalDistribution = meanAndVarianceForNormalDistributionForEachClusterForEachDimension.at(clusterIndex).at(dimensionIndex);
					dataUnitStruct.dataUnit =  normalDistributionRandomFloat(meanAndVarianceForNormalDistribution.mean,meanAndVarianceForNormalDistribution.variance);
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
	}

	//now we have to shuffle the data.
	std::srand ( unsigned ( std::time(0) ) );
	std::random_shuffle ( data.begin(), data.end() );


	std::string binaryFileName = fileName + ".dat";
	//now we have the data , we need to write it to file.
	char cstrFileName[binaryFileName.size() + 1];
	std::strcpy(cstrFileName, binaryFileName.c_str());

	//start by deleting the previus file
	std::remove(cstrFileName);
	FILE* file = fopen (cstrFileName, "wb");

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
	std::string metaDataFileName = "meta_data_" + fileName + ".txt";

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

