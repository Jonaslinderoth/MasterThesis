/*
 * arrayEqual.h
 *
 *  Created on: Mar 3, 2020
 *      Author: mikkel
 */

#ifndef ARRAYEQUAL_H_
#define ARRAYEQUAL_H_

#include <utility>
#include <vector>


bool areTheyEqual_d(bool* a_d , bool* b_d , unsigned long lenght);

bool areTheyEqual_d(unsigned int* a_d , unsigned int* b_d , unsigned long lenght);

bool printArray(unsigned int* a_d , unsigned long lenght , unsigned long maxHowMuch = 16);
bool printArray(bool* a_d , unsigned long lenght , unsigned long maxHowMuch = 16);

bool printPair_h(std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> a_h , const unsigned long maxLenght = 50);




std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> pointsContained(std::vector<std::vector<bool>*>* dims,
																					   std::vector<std::vector<float>*>* data,
																					   std::vector<unsigned int>* centroids,
																					   int m,
																					   float width = 10.0 ,
																					   unsigned long version = 0,
																					   unsigned long garbage = 0);

bool areTheyEqual_h(std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> a_h , std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> b_h , bool print = false);



#endif /* ARRAYEQUAL_H_ */
