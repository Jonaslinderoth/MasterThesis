/*
 * Cluster.h
 *
 *  Created on: Jan 29, 2020
 *      Author: mikkel
 */

#ifndef CLUSTER_H_
#define CLUSTER_H_


#include <limits>
#include <cstddef>
#include <vector>

enum DistributionType{
	uniformDistribution,
	normalDistribution,
	normalDistributionSpecial,
	constant
};
static const float lowerFloat = 0.0; //std::numeric_limits<float>::lowest();
static const float upperFloat = 100.0;   //std::numeric_limits<float>::max();

struct BoundsForUniformDistribution{
	float lower = lowerFloat;
	float upper = upperFloat;
};
struct MeanAndVarianceForNormalDistribution{
	float mean = 50;
	float variance = 15;
	unsigned int q = 1;
};


class Cluster {
public:
	Cluster();
	/*
	 * whatDimension is 0 indexed , the first dimension is 0! then 1 , then 2 ecc.
	 */
	bool addDimension(DistributionType distributionType = uniformDistribution,
			BoundsForUniformDistribution boundsForUniformDistribution = {lowerFloat, upperFloat},
			MeanAndVarianceForNormalDistribution meanAndVarianceForNormalDistribution = {50,15,1},
			float constant = 21,
			signed int whatDimension = -1);
	std::vector<float> getConstantForEachDimension();
	std::vector<DistributionType> getDistributionTypeForEachDimension();
	std::vector<BoundsForUniformDistribution> getBoundsForUniformDistributionForEachDimension();
	std::vector<MeanAndVarianceForNormalDistribution> getMeanAndVarianceForNormalDistributionForEachDimension();
	bool setAmmount(unsigned int ammount_);
	unsigned int getAmmount();
	bool setOutLierPercentage(float outLiers_);
	float getOutLierPercentage();
	virtual ~Cluster();
private:
	float outLierPercentage;
	unsigned int ammount;
	std::vector<float> constantForEachDimension;
	std::vector<DistributionType> distributionTypeForEachDimension;
	std::vector<BoundsForUniformDistribution> boundsForUniformDistributionForEachDimension;
	std::vector<MeanAndVarianceForNormalDistribution> meanAndVarianceForNormalDistributionForEachDimension;
};

#endif /* CLUSTER_H_ */
