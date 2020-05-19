#ifndef TERMINALCOMANDEXPERIMENT_H
#define TERMINALCOMANDEXPERIMENT_H
#include "Experiment.h"
#include <string>

class TerminalComandExperiment: public Experiment{
 public: TerminalComandExperiment(std::string command, std::string name, std::string dir): Experiment(2, name, dir, ""){
		this->command = command;
	};
	void start() override;

 private:
	std::string command = "";
};



#endif
