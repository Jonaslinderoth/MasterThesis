//
// Created by Jonas Linderoth on 20/04/2020.
//



#include <time.h>
#include <cmath>
#include <math.h>
#include "UI.h"
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <iomanip>
#include "termbox.h"
#include <algorithm>    // std::max
#include <thread>         // std::thread


void watcher(){
	std::cout << "thread started" << std::endl;
	unsigned int counter = 0;
	while(1) {
        struct tb_event ev;
        tb_poll_event(&ev);

        if (ev.key == TB_KEY_ESC) {
			counter++;
			if(counter >3){
				tb_shutdown();
				return;	
			}
        }else{
			counter =0;
		}
    }
}


UI::UI() {
    int code = tb_init();
    if (code < 0) {
        fprintf(stderr, "termbox init failed, code: %d\n", code);
        tb_select_input_mode(TB_INPUT_ESC);
        tb_select_output_mode(TB_OUTPUT_NORMAL);
        tb_clear();
    }
	std::thread* thr = new std::thread(watcher);
	
}


std::string getTimestamp() {
    // get a precise timestamp as a string
    const auto now = std::chrono::system_clock::now();
    const auto nowAsTimeT = std::chrono::system_clock::to_time_t(now);
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
																			 now.time_since_epoch()) % 1000;
    std::stringstream nowSs;
    nowSs << std::put_time(std::localtime(&nowAsTimeT), "%d/%m/%y %H:%M:%S");
	//	<< '.' << std::setfill('0') << std::setw(0) << nowMs.count();
    return nowSs.str();
}
std::string getShortTimestamp() {
    // get a precise timestamp as a string
    const auto now = std::chrono::system_clock::now();
    const auto nowAsTimeT = std::chrono::system_clock::to_time_t(now);
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
																			 now.time_since_epoch()) % 1000;
    std::stringstream nowSs;
    nowSs << std::put_time(std::localtime(&nowAsTimeT), "%H:%M:%S");
	//	<< '.' << std::setfill('0') << std::setw(0) << nowMs.count();
    return nowSs.str();
}

void UI::testDone(std::string title, Experiment* bar) {
	if (tb_width() < 200){
		this->testsDone.push_back("[ " + getShortTimestamp() + " ] " +title);		
	}else{
		this->testsDone.push_back("[ " + getTimestamp() + " ] " +title);
	}

    int id = this->progressBar.size();
    for(int i = 0; i < this->progressBar.size(); i++){
        if(this->progressBar.at(i) == bar){
            id = i;
            break;
        }
    }

    if(id>0){

		delete this->progressBar.at(id);
    }

    this->progressBar.erase(this->progressBar.begin()+id);
    this->updateProgressbar(this->progressBar.size());

    if(this->progressBar.size() == 0){
        this->done();
    }

}

void UI::addTestDone(std::string title){
	if (tb_width() < 200){
		this->testsDone.push_back("[ " + getShortTimestamp() + " ] " +title);		
	}else{
		this->testsDone.push_back("[ " + getTimestamp() + " ] " +title);
	}
}

void UI::addError(std::string error, std::string title) {
	std::string a;
	if (tb_width() < 200){
		a = "[ " + getShortTimestamp() + " ] " + error +" : "+ title;
	}else{
	    a = "[ " + getTimestamp() + " ] " + error +" : "+ title;	
	}
    this->errors.push_back(a);
}

void UI::updateProgressbar(unsigned int barNumber) {
    unsigned int start = 0;
    unsigned int end = this->progressBar.size();



    if(barNumber < this->progressBar.size()){
        if(this->updateLimit>0){
            auto now = std::chrono::steady_clock::now();
            std::chrono::milliseconds diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - this->lastUpdate);

            if(diff.count() <= this->updateLimit){
                return;
            }else{
                tb_clear();
            }
        }else{
            start = barNumber;
            end = barNumber+1;
        }


    }else{
        tb_clear();
    }
    this->lastUpdate = std::chrono::steady_clock::now();
    for(int j = start; j < end; j++){
        auto bar = this->progressBar.at(j);

        for(int i = 0; i < 20; i++){
            if(i < bar->getName().size()){
                tb_change_cell(i, j, bar->getName().at(i), TB_WHITE, TB_DEFAULT);
            }else{
                tb_change_cell(i, j, ' ', TB_WHITE, TB_DEFAULT);
            }
        }
        tb_change_cell(21, j, '|', TB_WHITE, TB_DEFAULT);

        unsigned int width = tb_width()-75;
        float fill  = ((float)bar->getTestsRan()/bar->getNumberOfTests())*width;
        for(int i = 0; i < width; i++){
            if(i <= fill){
                tb_change_cell(i+21, j, ' ', TB_WHITE, TB_GREEN);
            }else{
                tb_change_cell(i+21, j, ' ', TB_WHITE, TB_DEFAULT);
            }
        }

        auto now = std::chrono::steady_clock::now();
        std::chrono::nanoseconds diff = std::chrono::duration_cast<std::chrono::nanoseconds>(now - bar->getTimeStarted());


        double avgTime = (double)diff.count()/bar->getTestsRan();
        double eta = (bar->getNumberOfTests()-bar->getTestsRan())*avgTime;
        float pct  = ((float)bar->getTestsRan()/bar->getNumberOfTests())*100;
        char buffer[50];

        int ret = snprintf(buffer, sizeof(buffer), "| %5.1f %% | %6u / %6u | ", pct, bar->getTestsRan(), bar->getNumberOfTests());
        for(int i = 0; i < strlen(buffer); i++){
            tb_change_cell(width+21+3+i, j, buffer[i], TB_WHITE, TB_DEFAULT);
        }
        char* eta_s = this->timeToString(eta);
        for(int i = 0; i < strlen(eta_s); i++) {
            tb_change_cell(width+21 + 3 + strlen(buffer) + i, j, eta_s[i], TB_WHITE, TB_DEFAULT);
        }
        delete eta_s;

    }



	if(tb_height() < 1000){
		int mesH = tb_height()-this->progressBar.size()-2;
		unsigned int l = this->progressBar.size(); // l is the current height;
		unsigned int start = std::max((int)this->testsDone.size()-(mesH/2),(int)0);
		unsigned int end = this->testsDone.size(); 
		for(int k = start; k < end; k++){
			for(int f = 0; f < tb_width(); f++){
				if(f < this->testsDone.at(k).size()){
					tb_change_cell(f, l, this->testsDone.at(k).at(f), TB_GREEN, TB_DEFAULT);
				}else{
					tb_change_cell(f, l, ' ', TB_WHITE, TB_DEFAULT);
				}
			}
			l++;
		}
		unsigned int start2 = std::max((int)this->errors.size()-(mesH/2),(int)0);
		unsigned int end2 = this->errors.size(); 
		for(int k = start2; k < end2; k++){
			for(int f = 0; f < tb_width(); f++){
				if(f < this->errors.at(k).size()){
					tb_change_cell(f, l, this->errors.at(k).at(f), TB_RED, TB_DEFAULT);
				}else{
					tb_change_cell(f, l, ' ', TB_WHITE, TB_DEFAULT);
				}
			}
			l++;
		}		
		
	}else{
		int mesH = tb_height()-this->progressBar.size()-2;
		unsigned int messWidth1 = (tb_width()-20)/3;
		unsigned int l = this->progressBar.size()+2;
		unsigned int start2 = std::max((int)this->testsDone.size()-mesH,(int)0);
		unsigned int end2 = this->testsDone.size(); 
		for(int k = start2; k < end2; k++){
			for(int f = 0; f < messWidth1; f++){
				if(f < this->testsDone.at(k).size()){
					tb_change_cell(f, l, this->testsDone.at(k).at(f), TB_GREEN, TB_DEFAULT);
				}else{
					tb_change_cell(f, l, ' ', TB_WHITE, TB_DEFAULT);
				}
			}
			l++;
		}

		for(int i = 0; i < tb_width();i++){
			tb_change_cell(i, this->progressBar.size(), ' ', TB_GREEN, TB_DEFAULT);
		}
		for(int i = 0; i < tb_width();i++){
			tb_change_cell(i, this->progressBar.size()+1, ' ', TB_GREEN, TB_DEFAULT);
		}
		l = this->progressBar.size()+2;
		char mes[] = "Tests ran:";
		for(int i = 0; i < strlen(mes);i++){
			tb_change_cell(i, this->progressBar.size()+1, mes[i], TB_GREEN, TB_DEFAULT);
		}

		char mes2[] = "Errors:";
		for(int i = 0; i < strlen(mes2);i++){
			tb_change_cell(i+messWidth1+10, this->progressBar.size()+1, mes2[i], TB_RED, TB_DEFAULT);
		}
		start2 = std::max((int)this->errors.size()-mesH,(int)0);
		end2 = this->errors.size();
		for(int k = start2; k < end2; k++) {
			for (int f = 0; f < messWidth1*2; f++) {
				if (f < this->errors.at(k).size()) {
					tb_change_cell(f + messWidth1+10, l, this->errors.at(k).at(f), TB_RED, TB_DEFAULT);
				} else {
					tb_change_cell(f + messWidth1+10, l, ' ', TB_RED, TB_DEFAULT);
				}
			}
			l++;
		}
	}


    tb_present();
}

void UI::addProgressbar(Experiment *bar) {
    this->progressBar.push_back(bar);
    this->updateProgressbar(this->progressBar.size()-1);
}

void UI::updateProgressbar(Experiment *bar) {
    int id = this->progressBar.size();
    for(int i = 0; i < this->progressBar.size(); i++){
        if(this->progressBar.at(i) == bar){
            id = i;
            break;
        }
    }
    this->updateProgressbar(id);


}

char *UI::timeToString(double time) {
    char* buffer = new char[100];
    if(!isinf(time)){
		unsigned int sec = time/pow(10,9);
		unsigned int min = sec/60;
		unsigned int hr = min/60;
		min = min%60;
		sec = sec%60;

		int ret = snprintf(buffer, 100*sizeof(char), "| ETA: %2u:%2u:%2us |", hr, min, sec);

    }else{
        int ret = snprintf(buffer, 100*sizeof(char), "| ETA: ");
    }

    return buffer;
}

void UI::done() {
    char mes[] = "All experiments are done, press ESC to exit!";
    for(int i = 0; i < strlen(mes);i++){
        tb_change_cell(i, 0, mes[i], TB_BLACK, TB_WHITE);
    }
    tb_present();
    while(1) {
        struct tb_event ev;
        tb_poll_event(&ev);

        if (ev.key == TB_KEY_ESC) {
            tb_shutdown();
            return;
        }
    }
}


