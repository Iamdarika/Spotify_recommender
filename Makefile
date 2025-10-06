# High-Performance Music Recommendation Engine - Makefile
# Compiles C++ and CUDA source files with OpenMP and cuBLAS support

# Compiler settings
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -std=c++11 -O3 -Wall -fopenmp
NVCCFLAGS = -std=c++11 -O3 -arch=sm_75 -Xcompiler -fopenmp

# Include directories
INCLUDES = -I/usr/local/cuda/include

# Library directories and libraries
LDFLAGS = -L/usr/local/cuda/lib64
LDLIBS_BASE = -lgomp -ldl
LDLIBS_CUDA = -lcudart -lcublas

ifeq ($(DISABLE_CUDA),1)
	NVCCFLAGS += -DDISABLE_CUDA
	CXXFLAGS += -DDISABLE_CUDA
	LD_CUDA :=
	CUDA_NOTE = "[CPU-ONLY BUILD] CUDA disabled (DISABLE_CUDA=1)"
else
	LD_CUDA := $(LDLIBS_CUDA)
	CUDA_NOTE = "[GPU BUILD] CUDA enabled"
endif

LDLIBS = $(LD_CUDA) $(LDLIBS_BASE)

# Target executable
TARGET = recommender

# Source files
CPP_SOURCES = main.cpp DataManager.cpp
CU_SOURCES ?= Recommender.cu

# Object files
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
OBJECTS = $(CPP_OBJECTS) $(CU_OBJECTS)

# Default target
all: $(TARGET)

# Link all object files to create the executable
$(TARGET): $(OBJECTS)
	@echo $(CUDA_NOTE)
	@echo "Linking $@..."
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJECTS) $(LDFLAGS) $(LDLIBS)
	@echo "Build successful! Executable: $(TARGET)"

# Compile C++ source files
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA source files
%.o: %.cu
	@echo "Compiling $<..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(OBJECTS) $(TARGET) songs_data.bin
	@echo "Clean complete!"

# Clean only object files (keep executable and data)
clean-obj:
	@echo "Cleaning object files..."
	rm -f $(OBJECTS)
	@echo "Clean complete!"

# Run preprocessing example
preprocess: $(TARGET)
	@echo "Running preprocessing (requires dataset.csv)..."
	./$(TARGET) --preprocess dataset.csv

# Display help
help:
	@echo "Music Recommendation Engine - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build the recommender executable"
	@echo "  make all          - Same as 'make'"
	@echo "  make clean        - Remove all build artifacts and data files"
	@echo "  make clean-obj    - Remove only object files"
	@echo "  make preprocess   - Run preprocessing on dataset.csv"
	@echo "  make help         - Display this help message"
	@echo ""
	@echo "Usage:"
	@echo "  ./$(TARGET) --preprocess <csv_file>"
	@echo "  ./$(TARGET) --song \"Song Name\" [-n N]"
	@echo "  ./$(TARGET) --id \"track_id\" [-n N]"

# Phony targets
.PHONY: all clean clean-obj preprocess help

# Dependencies
main.o: main.cpp Song.h DataManager.h Recommender.h
DataManager.o: DataManager.cpp DataManager.h Song.h
Recommender.o: Recommender.cu Recommender.h Song.h
