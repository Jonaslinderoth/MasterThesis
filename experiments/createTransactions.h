#ifndef CREATETRANSACTIONS2_H
#define CREATETRANSACTIONS2_H
#include "Experiment.h"
class CreateTransactionsExperiments: public Experiment{
 public:
 CreateTransactionsExperiments(std::string name, std::string dir): Experiment(0,name, dir, "number of points, dim, version, passedTest, time"){
	}
	void start() override;
};


#endif
