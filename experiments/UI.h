//
// Created by Jonas Linderoth on 20/04/2020.
//

#ifndef UNTITLED2_UI_H
#define UNTITLED2_UI_H

#include <iostream>
#include <vector>
#include "Experiment.h"
class Experiment;
class UI{
    public:
        UI();
        void addProgressbar(Experiment* bar);
        void updateProgressbar(unsigned int barNumber);
        void updateProgressbar(Experiment* bar);
        void addError(std::string error, std::string name);
        void testDone(std::string title, Experiment* bar);
		void addTestDone(std::string title);
        char* timeToString(double time);
        void setTimeLimit(unsigned int millis){
            this->updateLimit = millis;
        }
        void done();
    private:
        std::vector<Experiment*> progressBar;
        std::vector<std::string> testsDone;
        std::vector<std::string> errors;
        unsigned int updateLimit = 100; //milliseconds
        std::chrono::steady_clock::time_point lastUpdate = std::chrono::steady_clock::now();
};


#endif //UNTITLED2_UI_H
