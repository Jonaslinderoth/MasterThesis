include config.mk
OS_NAME := $(shell uname -s | tr A-Z a-z)

ifeq ($(OS_NAME),darwin)
CXX = g++
CXXFLAGS=-I. -nocudalib -O2 #-g -G  
else
CXX = nvcc
CXXFLAGS=-I/usr/local/include/coin -I. -arch=sm_37 -DNDEBUG -O3 #-g -G #-DNDEBUG -O3 #-g -G # -DNDEBUG -g -G
#-lineinfo -maxrregcount 64 -lineinfo -maxrregcount 64 -g -G #
endif
#-maxrregcount 64 is needed for compiling the new merge support in debug mode, since there is not enough registers if some values are not optimised away

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


EXPERIMENTS = experiments
EXPERIMENTSDIR = experiments
EXPERIMENTSFILES = $(shell find $(EXPERIMENTSDIR) -name '*.cu') $(shell find $(EXPERIMENTSDIR) -name '*.cpp')
EXPERIMENTSOBJS= $(patsubst %.cu, %.o, $(patsubst %.cpp, %.o, $(EXPERIMENTSFILES)))


EXPERIMENTSDIR_ = experiments_exe
EXPERIMENTSFILES_ = $(shell find $(EXPERIMENTSDIR_) -name '*.cu') $(shell find $(EXPERIMENTSDIR_) -name '*.cpp')
EXPERIMENTSOBJS_ = $(patsubst %.cu, %.o, $(patsubst %.cpp, %.o, $(EXPERIMENTSFILES_)))


all: $(sources) $(EXE_DIR)/${EXE}  $(EXE_DIR)/${TEST}

os:
	echo $(OS_NAME)

.PHONY: all multi multi_test
multi:
	make -j4 all

multi_test: 
	make -j4 test

a:
	echo $(EXPERIMENTSFILES_)
	echo $(EXPERIMENTSOBJS_)

b: $(EXE_DIR)/experimentMineClusNaive
	echo $(DEPS)
	echo $(SOURCES)
	echo $(OBJECTS)	

test: $(EXE_DIR)/${TEST}
	./$(EXE_DIR)/${TEST} --gtest_filter=-*_SUPER_SLOW_*

experiments: $(EXE_DIR)/$(EXPERIMENTS) $(EXE_DIR)/experimentMineClusNaive $(EXE_DIR)/experimentMineClusNaiveMedium $(EXE_DIR)/experimentMineClusNaiveMnist $(EXE_DIR)/experimentFastDOCNaive $(EXE_DIR)/experimentFastDOCNaiveMedium $(EXE_DIR)/experimentDOCNaive $(EXE_DIR)/experimentDOCNaiveMedium $(EXE_DIR)/experimentMineClusBest $(EXE_DIR)/experimentMineClusBestMedium $(EXE_DIR)/experimentFastDOCBest $(EXE_DIR)/experimentFastDOCBestMedium $(EXE_DIR)/experimentDOCBest $(EXE_DIR)/experimentDOCBestMedium
	./$(EXE_DIR)/$(EXPERIMENTS)

benchmark: $(EXE_DIR)/$(BENCHMARK)
	./$(EXE_DIR)/$(BENCHMARK) --benchmark_repetitions=10 --benchmark_report_aggregates_only=true

test_fast: $(EXE_DIR)/${TEST}
	./$(EXE_DIR)/${TEST} --gtest_filter=-*SLOW*:*testClusteringPattern*

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
	mkdir -p $(BUILD_DIR)/$(EXPERIMENTSDIR)
	mkdir -p $(BUILD_DIR)/$(EXPERIMENTSDIR_)	
	${CXX} $(CXXFLAGS) -c $(LIBS)  $^ -o $@



#Target for test 	
$(EXE_DIR)/$(TEST): $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(addprefix $(BUILD_DIR)/, $(TESTOBJS))
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(addprefix $(BUILD_DIR)/, $(TESTOBJS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBGTEST) $(LIBS) -o $@

#Target for benchmark 	
$(EXE_DIR)/$(BENCHMARK): $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(addprefix $(BUILD_DIR)/, $(BENCHMARKOBJS))
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(addprefix $(BUILD_DIR)/, $(BENCHMARKOBJS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -lbenchmark -lbenchmark_main -o $@


#Target for experiments 	
$(EXE_DIR)/$(EXPERIMENTS): $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(addprefix $(BUILD_DIR)/, $(EXPERIMENTSOBJS))
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(addprefix $(BUILD_DIR)/, $(EXPERIMENTSOBJS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -ltermbox -o $@


$(EXE_DIR)/experimentMineClusNaive: $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(BUILD_DIR)/experiments_exe/experimentMineClusNaive.o
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/experiments_exe/experimentMineClusNaive.o $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -o $@


$(EXE_DIR)/experimentMineClusNaiveMedium: $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(BUILD_DIR)/experiments_exe/experimentMineClusNaiveMedium.o
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/experiments_exe/experimentMineClusNaiveMedium.o $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -o $@


$(EXE_DIR)/experimentMineClusNaiveMnist: $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(BUILD_DIR)/experiments_exe/experimentMineClusNaiveMnist.o
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/experiments_exe/experimentMineClusNaiveMnist.o $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -o $@


$(EXE_DIR)/experimentFastDOCNaive: $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(BUILD_DIR)/experiments_exe/experimentFastDOCNaive.o
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/experiments_exe/experimentFastDOCNaive.o $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -o $@


$(EXE_DIR)/experimentFastDOCNaiveMedium: $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(BUILD_DIR)/experiments_exe/experimentFastDOCNaiveMedium.o
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/experiments_exe/experimentFastDOCNaiveMedium.o $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -o $@

$(EXE_DIR)/experimentDOCNaive: $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(BUILD_DIR)/experiments_exe/experimentDOCNaive.o
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/experiments_exe/experimentDOCNaive.o $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -o $@


$(EXE_DIR)/experimentDOCNaiveMedium: $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(BUILD_DIR)/experiments_exe/experimentDOCNaiveMedium.o
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/experiments_exe/experimentDOCNaiveMedium.o $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -o $@



$(EXE_DIR)/experimentMineClusBest: $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(BUILD_DIR)/experiments_exe/experimentMineClusBest.o
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/experiments_exe/experimentMineClusBest.o $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -o $@


$(EXE_DIR)/experimentMineClusBestMedium: $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(BUILD_DIR)/experiments_exe/experimentMineClusBestMedium.o
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/experiments_exe/experimentMineClusBestMedium.o $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -o $@




$(EXE_DIR)/experimentFastDOCBest: $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(BUILD_DIR)/experiments_exe/experimentFastDOCBest.o
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/experiments_exe/experimentFastDOCBest.o $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -o $@


$(EXE_DIR)/experimentFastDOCBestMedium: $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(BUILD_DIR)/experiments_exe/experimentFastDOCBestMedium.o
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/experiments_exe/experimentFastDOCBestMedium.o $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -o $@

$(EXE_DIR)/experimentDOCBest: $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(BUILD_DIR)/experiments_exe/experimentDOCBest.o
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/experiments_exe/experimentDOCBest.o $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -o $@


$(EXE_DIR)/experimentDOCBestMedium: $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(BUILD_DIR)/experiments_exe/experimentDOCBestMedium.o
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/experiments_exe/experimentDOCBestMedium.o $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS) -o $@




clean:
	-rm -rf bin/
	-rm -rf build/
	-rm *.dat
	-rm *.txt
	-rm -rf testData



# to install coin-or and rehearce
# https://github.com/coin-or/Rehearse/blob/master/INSTALL
# global install: https://github.com/coin-or/Rehearse/issues/7
# 
# cannot find shared lib => sudo ldconfig


#install termbox
# https://github.com/nsf/termbox.git
