#ifndef MERGECANDIDATES_H
#define MERGECANDIDATES_H

#include "Experiment.h"
class MergeCandidatesExperiment: public Experiment{
 public:
 MergeCandidatesExperiment(std::string name, std::string dir, unsigned int memLimit): Experiment(0,name, dir, "number of points, dim, version, passedTest, time"){
		this->memLimit = memLimit;
	}
	void start() override;
 private:
	unsigned int memLimit = 10; // in gb
};


#endif
