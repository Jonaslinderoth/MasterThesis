#ifndef EXPERIMENT_H
#define EXPERIMENT_H
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>
#include "UI.h"

class UI;
class Experiment{
    public:
        std::chrono::steady_clock::time_point getTimeStarted(){
            return this->timeStarted;
        };

        std::chrono::steady_clock::time_point getTimeStopped(){
            return this->timeStopped;
        };
        unsigned int getNumberOfTests(){
            return this->numberOfTests;
        };

        unsigned int getTestsRan(){
            return this->testsRan;
        };
        bool isRunning(){
            return this->running;
        };
        virtual void start();

        void repportError(std::string error, std::string title);

        void setUI(UI* ui){
            this->ui = ui;
        }

        UI* getUI(){
            return this->ui;
        }

        std::string getName(){
            return this->name;
        }
        void setFileName(std::string fileName){
            this->fileName = fileName;
        }

        void setHeader(std::string header){
            this->header = header;
        }

        void writeLineToFile(std::string line);
		void testDone(std::string title);
        virtual ~Experiment();


    protected:
        Experiment(unsigned int numberOfTests, std::string name, std::string dir, std::string header){
            this->numberOfTests = numberOfTests;
            this->name = name;
            this->setHeader(header);
            this->setFileName(dir + "/" + name + ".csv");

            struct stat buffer;
            bool exists = (stat (this->fileName.c_str(), &buffer) == 0);


            this->file = std::ofstream(fileName, std::ios_base::app | std::ios_base::out);
            if(!exists){
            this->writeLineToFile(this->header);
            }
        };
        Experiment(unsigned int numberOfTests, std::string name){
            this->numberOfTests = numberOfTests;
            this->name = name;
        };
        virtual void stop();

        void testDone();
        void addTests(unsigned int numberOfTests){
            this->numberOfTests += numberOfTests;
        };

    private:
        std::chrono::steady_clock::time_point timeStarted;
        std::chrono::steady_clock::time_point timeStopped;
        unsigned int numberOfTests;
        unsigned int testsRan = 0;
        bool running = false;
        UI* ui = nullptr;
        std::string name;
        std::string fileName = "NONE";
        std::string header;
        std::ofstream file;
};
#endif
