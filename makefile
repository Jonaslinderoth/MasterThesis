include config.mk


CXX = nvcc

# -g -G for instrumntation for debugger
CXXFLAGS=-I. -arch=sm_37 -O2 #-g -G
LIBS = -lpthread

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




all: $(sources) $(EXE_DIR)/${EXE}  $(EXE_DIR)/${TEST}


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
	./$(EXE_DIR)/${TEST}

# Target for the main file defined in EXEFILE
$(EXE_DIR)/$(EXE): $(BUILD_DIR)/$(EXEFILE).o $(BUILD_DIR)/$(EXEFILE).d $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS))
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(BUILD_DIR)/$(EXEFILE).o  $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBS)



${BUILD_DIR}/%.d: %.cu main.cu
	mkdir -p $(BUILD_DIR)
	mkdir -p $(addprefix $(BUILD_DIR)/, $(shell find $(SOURCEDIR) -type d)) 
	$(CXX) -MT $(@:.d=.o) -M $(CXXFLAGS) $^ > $@

${BUILD_DIR}/%.d: %.cpp
	mkdir -p $(BUILD_DIR)
	mkdir -p $(addprefix $(BUILD_DIR)/, $(shell find $(SOURCEDIR) -type d)) 
	$(CXX) -MT $(@:.d=.o) -M $(CXXFLAGS) $^ > $@

${BUILD_DIR}/%.o: %.cu
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(SOURCEDIR)
	${CXX} $(CXXFLAGS) -c $^ -o $@

${BUILD_DIR}/%.o: %.cpp
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(TESTDIR)
	mkdir -p $(BUILD_DIR)/$(SOURCEDIR)
	${CXX} $(CXXFLAGS) -c $^ -o $@	

#Target for test 	
$(EXE_DIR)/$(TEST): $(addprefix $(BUILD_DIR)/, $(DEPS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(addprefix $(BUILD_DIR)/, $(TESTOBJS))
	mkdir -p $(EXE_DIR)
	$(CXX) $(CXXFLAGS) $(addprefix $(BUILD_DIR)/, $(TESTOBJS)) $(addprefix $(BUILD_DIR)/, $(OBJECTS)) $(LIBGTEST) $(LIBS) -o $@


clean:
	-rm -rf bin/
	-rm -rf build/
	-rm *.dat
	-rm *.txt

