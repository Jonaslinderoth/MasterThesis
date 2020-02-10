#include <iostream>
#include <chrono>
#include <random>

int main ()
{
  typedef std::chrono::high_resolution_clock myclock;
  myclock::time_point beginning = myclock::now();

  // obtain a seed from a user string:
  std::string str;
  std::cout << "Please, enter a seed: ";
  std::getline(std::cin,str);
  std::seed_seq seed1 (str.begin(),str.end());

  // obtain a seed from the timer
  myclock::duration d = myclock::now() - beginning;
  unsigned seed2 = d.count();

  std::mt19937 generator (seed1);   // mt19937 is a standard mersenne_twister_engine
  std::cout << "Your seed produced: " << generator() << std::endl;

  generator.seed (1);
  std::cout << "1: " << generator() << std::endl;

  generator.seed (1);
  std::cout << "2: " << generator() << std::endl;

  return 0;
}