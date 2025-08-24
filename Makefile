# Makefile for HPC Project
# Compiler
CC = gcc

# Compiler flags (enable OpenMP + optimizations)
CFLAGS = -fopenmp -O2 -Wall

# Target executable name
TARGET = hpc_project

# Source files
SRC = main.c

# Build rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

# Run program
run: $(TARGET)
	./$(TARGET)

# Clean build files
clean:
	rm -f $(TARGET)
