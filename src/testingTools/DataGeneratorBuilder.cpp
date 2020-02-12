/*
 * DataGeneratorBuilder.cpp
 *
 *  Created on: Jan 29, 2020
 *      Author: mikkel
 */

#include <src/testingTools/DataGeneratorBuilder.h>
#include <src/dataReader/DataReader.h>
#include <src/testingTools/MetaDataFileReader.h>

#include <cstring>
#include <chrono>
#include <fstream>

DataGeneratorBuilder::DataGeneratorBuilder() {
	dimension = 0;
	dimensionSet = false;
	fileName = "test1";
	// TODO Auto-generated constructor stub

}

DataGeneratorBuilder::~DataGeneratorBuilder() {
	// TODO Auto-generated destructor stub
}


bool DataGeneratorBuilder::setDimension(unsigned int dimension_){

	for(std::vector<Cluster>::iterator it=allClusters.begin() ; it != allClusters.end() ; ++it){
		if(it->getDistributionTypeForEachDimension().size() > dimension)
		{
			return false;
		}
	}
	dimension = dimension_;
	dimensionSet = true;
	return true;
}

bool DataGeneratorBuilder::addCluster(Cluster cluster){
	if(dimensionSet && cluster.getDistributionTypeForEachDimension().size() > dimension)
	{
		return false;
	}
	if(dimensionSet && cluster.getDistributionTypeForEachDimension().size() < dimension)
	{
		cluster.addDimension();
		return this->addCluster(cluster);
	}
	allClusters.push_back(cluster);
	return true;
}

bool DataGeneratorBuilder::setFileName(std::string fileName_)
{
	fileName = fileName_;
	return true;
}

bool DataGeneratorBuilder::build()
{
	unsigned int numberOfDimensions = 0;
	unsigned int numberOfClusters = 0;
	if(dimensionSet)
	{
		numberOfDimensions = dimensionSet;
		for(std::vector<Cluster>::iterator it = allClusters.begin() ; it != allClusters.end() ; ++it){
			if(it->getDistributionTypeForEachDimension().size() < dimension)
			{
				it->addDimension();
				return this->build();
			}
		}
	}else{
		unsigned int maxDim = 0;
		for(std::vector<Cluster>::iterator it = allClusters.begin() ; it != allClusters.end() ; ++it){
			if(it->getDistributionTypeForEachDimension().size()>maxDim)
			{
				maxDim = it->getDistributionTypeForEachDimension().size();
			}
		}
		for(std::vector<Cluster>::iterator it = allClusters.begin() ; it != allClusters.end() ; ++it){
			if(it->getDistributionTypeForEachDimension().size() < maxDim)
			{
				it->addDimension();
				return this->build();
			}
		}
		numberOfDimensions = maxDim;
	}
	//need to fix make the outLiers.
	unsigned int numberOfOutLiers = 0;
	Cluster outLiers;
	for(std::vector<Cluster>::iterator it = allClusters.begin() ; it != allClusters.end() ; ++it){
		unsigned int outLiersInThisCluster = 0;
		float outLierPercentage = it->getOutLierPercentage();
		//error checking.
		if(outLierPercentage > 100.0001){
			std::cout << "outlier percentage to big" << outLierPercentage << std::endl;
			outLierPercentage = 100;
		}
		if(outLierPercentage < -.0000001){
			std::cout << "negative outlier percentage" << outLierPercentage << std::endl;
			outLierPercentage = 0;
		}
		outLiersInThisCluster = it->getAmmount()*outLierPercentage/100;
		if(outLiersInThisCluster > 0){
			numberOfOutLiers += outLiersInThisCluster;
			it->setAmmount(it->getAmmount()-outLiersInThisCluster);
		}
	}
	BoundsForUniformDistribution outLiersBoundsForUniformDistribution;
	MeanAndVarianceForNormalDistribution outLiersMeanAndVarianceForNormalDistribution;
	outLiers.addDimension(uniformDistribution,outLiersBoundsForUniformDistribution,outLiersMeanAndVarianceForNormalDistribution,21,(numberOfDimensions-1));
	outLiers.setAmmount(numberOfOutLiers);
	if(numberOfOutLiers > 0){
		outLiers.setOutLierPercentage(100);
		allClusters.push_back(outLiers);
	}


	std::vector<std::vector<DistributionType>> distribuitionTypeForEachClusterForEachDimension;
	std::vector<std::vector<BoundsForUniformDistribution>> uniBoundsForEachClusterForEachDimension;
	std::vector<std::vector<MeanAndVarianceForNormalDistribution>> meanAndVarianceForNormalDistributionForEachClusterForEachDimension;
	std::vector<std::vector<float>> constantForEachClusterForEachDimension;
	std::vector<unsigned int> ammount;


	for(std::vector<Cluster>::iterator it = allClusters.begin() ; it != allClusters.end() ; ++it){
		distribuitionTypeForEachClusterForEachDimension.push_back(it->getDistributionTypeForEachDimension());
		uniBoundsForEachClusterForEachDimension.push_back(it->getBoundsForUniformDistributionForEachDimension());
		meanAndVarianceForNormalDistributionForEachClusterForEachDimension.push_back(it->getMeanAndVarianceForNormalDistributionForEachDimension());
		constantForEachClusterForEachDimension.push_back(it->getConstantForEachDimension());
		ammount.push_back(it->getAmmount());
	}





	numberOfClusters = allClusters.size();

	return spitFiles(fileName,
			numberOfDimensions ,
				numberOfClusters ,
				distribuitionTypeForEachClusterForEachDimension ,
				uniBoundsForEachClusterForEachDimension ,
				meanAndVarianceForNormalDistributionForEachClusterForEachDimension,
				constantForEachClusterForEachDimension,
				ammount);


}

bool DataGeneratorBuilder::spitFiles(std::string fileName ,
		const unsigned int numberOfDimensions ,
		const unsigned int numberOfClusters ,
		std::vector<std::vector<DistributionType>> distribuitionTypeForEachClusterForEachDimension,
		std::vector<std::vector<BoundsForUniformDistribution>> uniBoundsForEachClusterForEachDimension,
		std::vector<std::vector<MeanAndVarianceForNormalDistribution>> meanAndVarianceForNormalDistributionForEachClusterForEachDimension,
		std::vector<std::vector<float>> constantForEachClusterForEachDimension,
		std::vector<unsigned int> numberOfPointForEachCluster){

	//starting by figureing out if it needs to be spit.


	unsigned int totalSize = 0;
	for(std::vector<unsigned int>::iterator iter = numberOfPointForEachCluster.begin() ; iter != numberOfPointForEachCluster.end() ; ++iter ){
		totalSize += *iter;
	}
	float sequrityFactor = 2; //magic number tested on our system
	float bytesInFloat = 4;
	float memoryUse = totalSize*numberOfDimensions*bytesInFloat*sequrityFactor;
	if(memoryUse < 0){   //200000000
		DataGenerator DG = DataGenerator(fileName,
				numberOfDimensions,
				numberOfClusters,
				distribuitionTypeForEachClusterForEachDimension,
				uniBoundsForEachClusterForEachDimension,
				meanAndVarianceForNormalDistributionForEachClusterForEachDimension,
				constantForEachClusterForEachDimension,
				numberOfPointForEachCluster);
		return true;
	}
	//std::cout << "divide and conquer case" << std::endl;
	//ok now the data is going to be too big, its not good because DataGenerator is not having a good time with that.
	//to solve that problem we spit the data in many files
	//and the combine them.

	//checking if one of the clusters is too big
	for(std::vector<unsigned int>::iterator iter = numberOfPointForEachCluster.begin() ; iter != numberOfPointForEachCluster.end() ; ++iter ){
		if(*iter*bytesInFloat*sequrityFactor*numberOfDimensions > 200000000){
			//std::cout << "one of the clusters is too big, make two that are equals" << std::endl;
		}
	}


	//now we make a file for each cluster
	//we need to store the names of the files
	std::vector<std::string> fileNamesOfMomentaryFiles;
	for(int clusterIndex = 0 ; clusterIndex < numberOfClusters ; clusterIndex++){
		//making new names
		std::string momentartyFileName = "_"+ std::to_string(clusterIndex)+ "_" + fileName;
		//storing new names
		fileNamesOfMomentaryFiles.push_back(momentartyFileName);
		std::vector<std::vector<DistributionType>> momentaryDistribuitionTypeForEachClusterForEachDimension;
		std::vector<std::vector<BoundsForUniformDistribution>> momentaryUniBoundsForEachClusterForEachDimension;
		std::vector<std::vector<MeanAndVarianceForNormalDistribution>> momentaryMeanAndVarianceForNormalDistributionForEachClusterForEachDimension;
		std::vector<std::vector<float>> momentaryConstantForEachClusterForEachDimension;
		std::vector<unsigned int> momentaryNumberOfPointForEachCluster;

		momentaryDistribuitionTypeForEachClusterForEachDimension.push_back(distribuitionTypeForEachClusterForEachDimension.at(clusterIndex));
		momentaryUniBoundsForEachClusterForEachDimension.push_back(uniBoundsForEachClusterForEachDimension.at(clusterIndex));
		momentaryMeanAndVarianceForNormalDistributionForEachClusterForEachDimension.push_back(meanAndVarianceForNormalDistributionForEachClusterForEachDimension.at(clusterIndex));
		momentaryConstantForEachClusterForEachDimension.push_back(constantForEachClusterForEachDimension.at(clusterIndex));
		momentaryNumberOfPointForEachCluster.push_back(numberOfPointForEachCluster.at(clusterIndex));

		DataGenerator DG = DataGenerator(momentartyFileName,
						numberOfDimensions,
						1,
						momentaryDistribuitionTypeForEachClusterForEachDimension,
						momentaryUniBoundsForEachClusterForEachDimension,
						momentaryMeanAndVarianceForNormalDistributionForEachClusterForEachDimension,
						momentaryConstantForEachClusterForEachDimension,
						momentaryNumberOfPointForEachCluster,
						clusterIndex);
	}

	//now we have the files spit , now we need to merge them
	//we need to read from the files , we need dataReaders
	std::vector<DataReader*> vectorOfDataReader;
	std::vector<unsigned int> vectorOfSizesOfFiles;
	std::vector<MetaDataFileReader*> vectorOfMetaDataFileReaders;
	unsigned int totalSizeTwo = 0;

	for(std::vector<std::string>::iterator iter = fileNamesOfMomentaryFiles.begin() ; iter != fileNamesOfMomentaryFiles.end() ; ++iter){
		unsigned int queueSize = 8;
		unsigned int heuristicMexMemoryUsageByAllReaders = 1600000000;
		//500 is just magic number
		unsigned int chunckSize = heuristicMexMemoryUsageByAllReaders / (numberOfClusters*queueSize*500);

		DataReader* dr = new DataReader(*iter,queueSize,chunckSize);
		vectorOfDataReader.push_back(dr);
		vectorOfSizesOfFiles.push_back(dr->getSize());
		totalSizeTwo += dr->getSize();
		MetaDataFileReader* mdfr = new MetaDataFileReader(*iter);
		vectorOfMetaDataFileReaders.push_back(mdfr);
	}

	if(totalSizeTwo != totalSize){
		std::cout << "the generator is bugged, spitting in files is not working" << std::endl;
	}else{
		//std::cout << "datGeneratorBuilder is merging the files" << std::endl;
	}

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
	float fsize = (float)totalSize;
	fwrite(&fsize, sizeof(float), 1, file);

	//making the meta data file also
	std::string metaDataFileName = "meta_data_" + fileName + ".txt";
	char cstrMetaDataFileName[metaDataFileName.size() + 1];
	std::strcpy(cstrMetaDataFileName, metaDataFileName.c_str());
	std::remove(cstrMetaDataFileName);
	std::ofstream metaDataOutfile;
	metaDataOutfile.open(metaDataFileName);

	metaDataOutfile << std::to_string(numberOfDimensions) << std::endl;
	metaDataOutfile << std::to_string(numberOfClusters) << std::endl;

	for(std::vector<MetaDataFileReader*>::iterator iter = vectorOfMetaDataFileReaders.begin() ; iter != vectorOfMetaDataFileReaders.end() ; ++iter){
		std::vector<std::string> clusterLines = (*iter)->getClusterLines();
		for(std::vector<std::string>::iterator linesIter = clusterLines.begin() ; linesIter != clusterLines.end() ; ++linesIter){
			metaDataOutfile << *linesIter << std::endl;
		}
	}



	for(int pointIndex = 0 ; pointIndex < totalSize ; ++pointIndex){
		unsigned goo = pointIndex;
		//if(goo%400000 == 0){
		//	std::cout << pointIndex << std::endl;
		//}

		unsigned int chosenClusterIndex = 0;
		unsigned int randomNumber = rand() % (totalSize-pointIndex);
		for(int clusterIndex = 0 ; clusterIndex < numberOfClusters ; ++clusterIndex){
			if(randomNumber < vectorOfSizesOfFiles.at(clusterIndex)){
				chosenClusterIndex = clusterIndex;
				//std::cout << "vectorOfSizesOfFiles" << std::endl;
				//std::cout << vectorOfSizesOfFiles.size() << std::endl;
				//std::cout << chosenClusterIndex << std::endl;
				vectorOfSizesOfFiles.at(chosenClusterIndex) -= 1;
				//we need to delete those temporary files

				if(vectorOfSizesOfFiles.at(chosenClusterIndex) == 0){
					char cstrMomentaryFileName[fileNamesOfMomentaryFiles.at(chosenClusterIndex).size() + 1];
					std::remove(cstrMomentaryFileName);
				}
				break;
			}
			randomNumber -= vectorOfSizesOfFiles.at(clusterIndex);
		}

		if(!vectorOfDataReader.at(chosenClusterIndex)->isThereANextPoint()){
			std::cout << "generator trying to get point from empty data reader" << std::endl;
		}else{

			MetaDataFileReader* mdfr = vectorOfMetaDataFileReaders.at(chosenClusterIndex);
			DataReader* dr = vectorOfDataReader.at(chosenClusterIndex);
			std::vector<float>* point = dr->nextPoint();
			for(std::vector<float>::iterator iter = point->begin() ; iter != point->end() ; ++iter){
				float f = *iter;
				fwrite(&f, sizeof(float), 1, file);
			}
			delete point;
			metaDataOutfile << mdfr->nextCheat() << std::endl;
		}
	}
	for(std::vector<DataReader*>::iterator iter = vectorOfDataReader.begin() ; iter != vectorOfDataReader.end() ; ++iter){
		delete *iter;
	}
	fclose(file);

	deleteFiles(fileNamesOfMomentaryFiles);

}

bool DataGeneratorBuilder::deleteFiles(std::vector<std::string> vecOfFilesNames){
	for(std::vector<std::string>::iterator iter = vecOfFilesNames.begin() ; iter != vecOfFilesNames.end() ; ++iter){
		std::string binaryFileName = *iter + ".dat";
		//now we have the data , we need to write it to file.
		char cstrFileName[binaryFileName.size() + 1];
		std::strcpy(cstrFileName, binaryFileName.c_str());

		//start by deleting the previus file
		std::remove(cstrFileName);
	}




	for(std::vector<std::string>::iterator iter = vecOfFilesNames.begin() ; iter != vecOfFilesNames.end() ; ++iter){
		std::string binaryFileName = "meta_data_" + *iter + ".txt";
		//now we have the data , we need to write it to file.
		char cstrFileName[binaryFileName.size() + 1];
		std::strcpy(cstrFileName, binaryFileName.c_str());

		//start by deleting the previus file
		std::remove(cstrFileName);
	}


return true;
}

bool DataGeneratorBuilder::buildUClusters(unsigned int ammountOfPoint,
		unsigned int ammountOfClusters,
		unsigned int with,
		unsigned int dimensions,
		unsigned dimensionUsed,
		float outLiersPersentage) {
	if(ammountOfClusters == 0){
		return false;
	}

	//need to calculate the points in each cluster , and because of integer division i need to take care of the rest.
	std::vector<unsigned int> ammountOfPointPerCluster;
	unsigned int pointsPerCluster = ammountOfPoint/ammountOfClusters;
	for(int clusterIndex = 0 ; clusterIndex < ammountOfClusters ; ++clusterIndex){
		ammountOfPointPerCluster.push_back(ammountOfPoint);
	}
	ammountOfPointPerCluster.at(0) += ammountOfPoint % ammountOfClusters;


	std::vector<Cluster> vecOfClusters;
	for(int clusterIndex = 0 ; clusterIndex < ammountOfClusters ; ++clusterIndex){
		Cluster c;
		c.setOutLierPercentage(outLiersPersentage);
		c.setAmmount(ammountOfPointPerCluster.at(clusterIndex));
		vecOfClusters.push_back(c);
	}


	//i need to pick with dimension to "work with".
	std::vector<int> vecOfDimensions;
	for(int i = 0 ; i < dimensions ; ++i){
		vecOfDimensions.push_back(i);
	}

	//check that we have enoth dimensions to work with
	if(dimensionUsed >= dimensions){
		return false;
	}
	DataGeneratorBuilder dgb;


	//pick the dimensions
	std::vector<int> dimensionChosen;
	for(int i = 0 ; i < dimensionUsed ; ++i){
		unsigned int chosen = (unsigned int)rand()%vecOfDimensions.size();
		dimensionChosen.push_back(vecOfDimensions.at(chosen));
		vecOfDimensions.erase(vecOfDimensions.begin()+chosen);
	}


	std::vector<BoundsForUniformDistribution> previusClusterBounds;
	for(int i = 0 ; i < dimensionChosen.size() ; ++i){
		BoundsForUniformDistribution boundsForUniformDistribution;
		int maxRange = ((int)boundsForUniformDistribution.upper-(int)boundsForUniformDistribution.lower-with);
		int lowerRange = ((int)rand()%maxRange)+(int)boundsForUniformDistribution.lower;
		int upperRange = lowerRange + with;
		boundsForUniformDistribution.lower = lowerRange;
		boundsForUniformDistribution.upper = upperRange;
		previusClusterBounds.push_back(boundsForUniformDistribution);
	}

	float variace = ((float)with)/12;

	for(std::vector<Cluster>::iterator cluster = vecOfClusters.begin() ; cluster != vecOfClusters.end() ; ++cluster){
		unsigned int dimensionIndex = 0;
		BoundsForUniformDistribution basicBoundsForUniformDistribution;
		MeanAndVarianceForNormalDistribution basicMeanAndVarianceForNormalDistribution;
		cluster->addDimension(uniformDistribution,basicBoundsForUniformDistribution,basicMeanAndVarianceForNormalDistribution,21,dimensions-1);
		for(std::vector<int>::iterator dim = dimensionChosen.begin() ; dim != dimensionChosen.end() ; ++dim){
			int lowerRange = 0;
			int upperRange = 0;
			if((int)rand()%2 == 0){
				lowerRange = previusClusterBounds.at(dimensionIndex).lower;
				upperRange = previusClusterBounds.at(dimensionIndex).upper;
				if((int)rand()%2 == 0){
					variace = -variace;
				}
				lowerRange += variace;
				upperRange += variace;
				if(lowerRange < basicBoundsForUniformDistribution.lower or upperRange > basicBoundsForUniformDistribution.upper){
					lowerRange -= (2*variace);
					upperRange -= (2*variace);
				}
			}else{
				vecOfDimensions.push_back(*dim);
				unsigned int chosen = (unsigned int)rand()%vecOfDimensions.size();
				*dim = (vecOfDimensions.at(chosen));
				vecOfDimensions.erase(vecOfDimensions.begin()+chosen);
				BoundsForUniformDistribution boundsForUniformDistribution;
				int maxRange = ((int)boundsForUniformDistribution.upper-(int)boundsForUniformDistribution.lower-with);
				lowerRange = ((int)rand()%maxRange)+(int)boundsForUniformDistribution.lower;
				upperRange = lowerRange + with;
			}

			//now set the range in the struct
			BoundsForUniformDistribution boundsForUniformDistribution;
			boundsForUniformDistribution.lower = lowerRange;
			boundsForUniformDistribution.upper = upperRange;
			previusClusterBounds.at(dimensionIndex) = boundsForUniformDistribution;
			cluster->addDimension(uniformDistribution,boundsForUniformDistribution,basicMeanAndVarianceForNormalDistribution,21,*dim);
			dimensionIndex++;
		}
		dgb.addCluster(*cluster);
	}
	dgb.build();
	return true;
}