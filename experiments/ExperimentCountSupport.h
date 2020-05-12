#ifndef EXPERIMENTCOUNTSUPPORT_H
#define EXPERIMENTCOUNTSUPPORT_H

#include "Experiment.h"
class ExperimentCountSupport: public Experiment{
 public:
 ExperimentCountSupport(std::string name, std::string dir): Experiment(0,name, dir, "number Of Candidates, number of points, dim, version, passedTest, time"){
	}
	void start() override;
};

#endif
