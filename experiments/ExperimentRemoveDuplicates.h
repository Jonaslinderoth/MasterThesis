#ifndef EXPERIMENTREMOVEDUPLICATES_H
#define EXPERIMENTREMOVEDUPLICATES_H

#include "Experiment.h"
class ExperimentRemoveDuplicates: public Experiment{
 public:
 ExperimentRemoveDuplicates(std::string name, std::string dir): Experiment(0,name, dir, "number Of Candidates, dim, version, passedTest, time"){
	}
	void start() override;
};

#endif
