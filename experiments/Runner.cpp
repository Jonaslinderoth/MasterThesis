#include "Runner.h"
#include <iostream>

void Runner::start(){
    if(this->getUI() == nullptr){
        this->setUI(new UI());
    }
	Experiment::start();
	for(unsigned int i = 0; i < this->experiments.size(); i++){
        this->experiments.at(i)->setUI(this->getUI());
		this->experiments.at(i)->start();
		while(this->experiments.at(i)->isRunning()){
			std::cout << "waiting..." << std::endl;
		}
		this->testDone();
	}
	this->stop();
}

void Runner::stop(){
	Experiment::stop();
}

void Runner::addExperiment(Experiment* newExperiment){
	this->addTests(1); // maybe this should just be +1? newExperiment->getNumberOfTests()
	this->experiments.push_back(newExperiment);
}

