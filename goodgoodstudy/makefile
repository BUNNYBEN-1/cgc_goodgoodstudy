CC = g++
CFLAGS = -Wall -std=c++11 -march=native -fopenmp
TARGET = ../goodgoodstudy.exe
SRC = source_code.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)