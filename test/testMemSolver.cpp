#include <gtest/gtest.h>
#include "../src/DOC_GPU/MemSolver.h"
TEST(testMemSolver, testSetup){
	MemSolver::computeForAllocations(1,1,1,1,1,1);
}
