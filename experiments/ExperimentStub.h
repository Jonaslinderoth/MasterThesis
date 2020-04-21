#ifndef EXPERIMENTSTUB_H
#define EXPERIMENTSTUB_H
#include "Experiment.h"

class ExperimentStub: public Experiment{
 public:
 ExperimentStub(unsigned int numberOfTests, unsigned int delay, std::string name, std::string dir): Experiment(numberOfTests, name, dir, "header"){
		this->delay = delay;
		//this->addTests(numberOfTests);
	};
	void start() override;
 private:
	unsigned int delay = 0;
};

#endif
