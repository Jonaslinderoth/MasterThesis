#ifndef MERGECANDIDATES22_H
#define MERGECANDIDATES22_H

#include "Experiment.h"
class MergeCandidatesExperimentSmem: public Experiment{
 public:
 MergeCandidatesExperimentSmem(std::string name, std::string dir, unsigned int memLimit): Experiment(0,name, dir, "number of points, dim, chunkSize, time"){
		this->memLimit = memLimit;
	}
	void start() override;
 private:
	unsigned int memLimit = 10; // in gb
};


#endif
