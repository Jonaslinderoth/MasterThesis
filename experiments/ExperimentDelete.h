#ifndef EXPERIMENTDELETE_H
#define EXPERIMENTDELETE_H

#include "Experiment.h"

class ExperimentDelete: public Experiment{
 public:
 ExperimentDelete(std::string name, std::string dir): Experiment(0, name, dir, "Number of Points, Dim, Delete fraction, Version, Time"){
		//this->addTests(numberOfTests);
	};
	void start() override;
 private:
};


#endif
