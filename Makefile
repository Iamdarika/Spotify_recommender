CC = gcc
NVCC = nvcc
CFLAGS = -O2 -I./src
SRC = src/main.c src/benchmark.c src/preprocess.c src/matrix_factorization.c
CU = src/collaborative_filtering.cu
OBJ = $(SRC:.c=.o)
TARGET = hpc_project

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(OBJ) $(CU) -o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)
