CC = gcc
NVCC = nvcc
CFLAGS = -O2 -I./src
SRC = src/benchmark.c src/preprocess.c src/matrix_factorization.c
CU = src/collaborative_filtering.cu
OBJ = $(SRC:.c=.o)
TARGET = hpc_project

all: $(TARGET)

$(TARGET): main.o $(OBJ)
	$(NVCC) main.o $(OBJ) $(CU) -o $(TARGET)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c -o main.o

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJ) main.o $(TARGET)
