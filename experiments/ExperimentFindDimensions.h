#ifndef EXPERIMENTFINDDIMENSIONS_H
#define EXPERIMENTFINDDIMENSIONS_H

#include "Experiment.h"
class ExperimentFindDimensions: public Experiment{
 public:
 ExperimentFindDimensions(std::string name, std::string dir): Experiment(0,name, dir, "Number of Points, dim, version, passedTest, time"){};
	void start() override;
};

#endif
