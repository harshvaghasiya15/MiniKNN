# MiniKNN Makefile
# Build:  make
# Run:    ./miniknn          (run from the project root so data/iris.csv is found)
# Clean:  make clean

CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude
TARGET   := miniknn
SOURCES  := $(wildcard src/*.cpp)

$(TARGET): $(SOURCES) $(wildcard include/*.h)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) $(TARGET).exe
