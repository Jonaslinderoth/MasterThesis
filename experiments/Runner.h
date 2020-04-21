#ifndef RUNNER_H
#define RUNNER_H
#include <time.h>
#include <vector>
#include "Experiment.h"
#include "UI.h"
class Runner: public Experiment{
 public:
 Runner(std::string name): Experiment(0, name){
 };
	void addExperiment(Experiment* experiment);
	void start() override;
	void stop() override;
 private:
	std::vector<Experiment*> experiments = std::vector<Experiment*>();
    Experiment* currentExperiment;
};

#endif
