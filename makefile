include config.mk
OS_NAME := $(shell uname -s | tr A-Z a-z)

ifeq ($(OS_NAME),darwin)
CXX = g++
CXXFLAGS=-I. -nocudalib -O2 #-g -G  
else
CXX = nvcc
CXXFLAGS=-I/usr/local/include/coin -I. -arch=sm_37 -O2 #-g -G 
endif

# -g -G for instrumntation for debugger, these might cause curand to not generate random numbers properly...
# -DNDEBUG will remove assertions

LIBS = -lpthread -lcurand -L/usr/local/cuda-10.2/lib64 -lCoinUtils -lClp  -lClpSolver -lrehearse -lOsiClp -lCbc -lOsiCbc

EXE=main
EXEFILE = main

SOURCEDIR = src
SOURCES=$(shell find $(SOURCEDIR) -name '*.cu') $(shell find $(SOURCEDIR) -name '*.cpp')
OBJECTS= $(patsubst %.cu, %.o, $(patsubst %.cpp, %.o, $(SOURCES)))
DEPS=$(patsubst %.cu, %.d, $(patsubst %.cpp, %.d, $(SOURCES)))

BUILD_DIR=build
EXE_DIR=bin

TEST = test
TESTDIR = test
TESTFILES = $(shell find $(TESTDIR) -name '*.cu') $(shell find $(TESTDIR) -name '*.cpp')
TESTOBJS= $(patsubst %.cu, %.o, $(patsubst %.cpp, %.o, $(TESTFILES)))

BENCHMARK = benchmark
BENCHMARKDIR = benchmark
BENCHMARKFILES = $(shell find $(BENCHMARKDIR) -name '*.cu') $(shell find $(BENCHMARKDIR) -name '*.cpp')
BENCHMARKOBJS= $(patsubst %.cu, %.o, $(patsubst %.cpp, %.o, $(BENCHMARKFILES)))



all: $(sources) $(EXE_DIR)/${EXE}  $(EXE_DIR)/${TEST}


os:
	echo $(OS_NAME)

.PHONY: all multi multi_test
multi:
	make -j4 all

multi_test: 
	make -j4 test
b: 
	echo $(DEPS)
	echo $(SOURCES)
	echo $(OBJECTS)	

test: $(EXE_DIR)/${TEST}
	./$(EXE_DIR)/${TEST} --gtest_filter=-*_SUPER_SLOW_*:**


benchmark: $(EXE_DIR)/$(BENCHMARK)
	./$(EXE_DIR)/$(BENCHMARK) --benchmark_repetitions=10 --benchmark_report_aggregates_only=true

test_fast: $(EXE_DIR)/${TEST}
	./$(EXE_DIR)/${TEST} --gtest_filter=-*_SLOW*:*testClusteringPattern*

# Target for the main file defined in EXEFILE
$(EXE_DIR)/$(EXE): $(BUILD_DIR)/$(EXEFILE).o $(BUILD_DIR)/$(EXEFILE).d $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS))
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(BUILD_DIR)/$(EXEFILE).o  $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS)



${BUILD_DIR}/%.d: %.cu main.cu
	mkdir -p $(BUILD_DIR)
	mkdir -p $(addprefix $(BUILD_DIR)/, $(shell find $(SOURCEDIR) -type d)) 
	$(CXX) -MT $(@:.d=.o) -M $(CXXFLAGS) $(LIBS) $^ > $@ 

${BUILD_DIR}/%.d: %.cpp
	mkdir -p $(BUILD_DIR)
	mkdir -p $(addprefix $(BUILD_DIR)/, $(shell find $(SOURCEDIR) -type d)) 
	$(CXX) -MT $(@:.d=.o) -M $(CXXFLAGS) $(LIBS) $^ > $@  

${BUILD_DIR}/%.o: %.cu
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(SOURCEDIR)
	${CXX} $(CXXFLAGS) -c  $(LIBS) $^ -o $@ 

${BUILD_DIR}/%.o: %.cpp
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(TESTDIR)
	mkdir -p $(BUILD_DIR)/$(SOURCEDIR)
	mkdir -p $(BUILD_DIR)/$(BENCHMARKDIR)	
	${CXX} $(CXXFLAGS) -c $(LIBS)  $^ -o $@



#Target for test 	
$(EXE_DIR)/$(TEST): $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(addprefix $(BUILD_DIR)/, $(TESTOBJS))
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(addprefix $(BUILD_DIR)/, $(TESTOBJS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBGTEST) $(LIBS) -o $@

#Target for benchmark 	
$(EXE_DIR)/$(BENCHMARK): $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(addprefix $(BUILD_DIR)/, $(BENCHMARKOBJS))
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(addprefix $(BUILD_DIR)/, $(BENCHMARKOBJS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -lbenchmark -lbenchmark_main -o $@



clean:
	-rm -rf bin/
	-rm -rf build/
	-rm *.dat
	-rm *.txt



# to install coin-or and rehearce
# https://github.com/coin-or/Rehearse/blob/master/INSTALL
# global install: https://github.com/coin-or/Rehearse/issues/7
# 
# cannot find shared lib => sudo ldconfig
