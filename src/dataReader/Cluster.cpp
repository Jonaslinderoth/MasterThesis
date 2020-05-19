/*
 * Cluster.cpp
 *
 *  Created on: Jan 29, 2020
 *      Author: mikkel
 */

#include <src/dataReader/Cluster.h>
#include <iostream>

Cluster::Cluster() {
	ammount = 128;
	outLierPercentage = 0.0;
	// TODO Auto-generated constructor stub

}


Cluster::~Cluster() {
	// TODO Auto-generated destructor stub
}

bool Cluster::setAmmount(unsigned int ammount_)
{
	ammount = ammount_;
	return true;
}

unsigned int Cluster::getAmmount()
{
	return ammount;
}


bool Cluster::addDimension(DistributionType distributionType,
		BoundsForUniformDistribution boundsForUniformDistribution,
		MeanAndVarianceForNormalDistribution meanAndVarianceForNormalDistribution,
		float constant,
		signed int whatDimension){
	if(whatDimension < -1){
		std::cout << "can not have negative dimension when building cluster" << std::endl;
		throw 1;
	}
	if((whatDimension == -1) or (whatDimension == distributionTypeForEachDimension.size()))
	{
		distributionTypeForEachDimension.push_back(distributionType);
		boundsForUniformDistributionForEachDimension.push_back(boundsForUniformDistribution);
		meanAndVarianceForNormalDistributionForEachDimension.push_back(meanAndVarianceForNormalDistribution);
		constantForEachDimension.push_back(constant);
	}else if(whatDimension > distributionTypeForEachDimension.size()){
		this->addDimension();
		this->addDimension(distributionType,boundsForUniformDistribution,meanAndVarianceForNormalDistribution,constant,whatDimension);
	}else{
		distributionTypeForEachDimension.at(whatDimension) = distributionType;
		boundsForUniformDistributionForEachDimension.at(whatDimension)=boundsForUniformDistribution;
		meanAndVarianceForNormalDistributionForEachDimension.at(whatDimension) = meanAndVarianceForNormalDistribution;
		constantForEachDimension.at(whatDimension) = constant;
	}
	return true;
}

std::vector<float> Cluster::getConstantForEachDimension()
{
	return constantForEachDimension;
}


std::vector<DistributionType> Cluster::getDistributionTypeForEachDimension(){
	return distributionTypeForEachDimension;
}

std::vector<BoundsForUniformDistribution> Cluster::getBoundsForUniformDistributionForEachDimension(){
	return boundsForUniformDistributionForEachDimension;
}

std::vector<MeanAndVarianceForNormalDistribution> Cluster::getMeanAndVarianceForNormalDistributionForEachDimension()
{
	return meanAndVarianceForNormalDistributionForEachDimension;
}

bool Cluster::setOutLierPercentage(float outLiers_) {
	outLierPercentage = outLiers_;
	return true;
}

float Cluster::getOutLierPercentage() {
	return outLierPercentage;
}
