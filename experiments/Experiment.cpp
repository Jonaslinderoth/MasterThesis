#include "Experiment.h"
#include <chrono>

void Experiment::testDone() {
    this->testsRan++;
    if(this->ui != nullptr){
        this->ui->updateProgressbar(this);
    }
}

void Experiment::testDone(std::string title) {
    this->testsRan++;
    if(this->ui != nullptr){
        this->ui->updateProgressbar(this);
		this->getUI()->addTestDone(this->name + " " + title);
    }
}

void Experiment::start() {
    if(this->ui != nullptr){
        this->ui->addProgressbar(this);
    }
    this->running = true;
    this->timeStarted = std::chrono::steady_clock::now();
}

void Experiment::stop() {
    this->running = false;
    this->timeStopped = std::chrono::steady_clock::now();
    this->getUI()->testDone(this->name, this);
}

void Experiment::repportError(std::string error, std::string title) {
    this->getUI()->addError(error, title);
}

Experiment::~Experiment() {
    this->file.close();
}

void Experiment::writeLineToFile(std::string line) {
    if(this->fileName == "NONE"){
        if(this->getUI() != nullptr){
            this->getUI()->addError("Trying to write to unspecified file", this->name);
        }
    }else{
        this->file << line << std::endl;
    }
}
