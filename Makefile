# Makefile for Monostatic RCS Sweep - Shooting & Bouncing Rays

# Compiler
NVCC = nvcc
CXX = g++

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Target executable
TARGET = $(BIN_DIR)/SagittaSBR

# Compiler flags
NVCC_FLAGS = -I$(INC_DIR) -arch=sm_75 -std=c++14
CXX_FLAGS = -I$(INC_DIR) -std=c++14

# Optimization flags (default: optimized build)
OPT_FLAGS = -O3 -use_fast_math

# Debug flags (use with 'make debug')
DEBUG_FLAGS = -g -G -O0 -DDEBUG

# Profiling flags (use with 'make profile')
PROFILE_FLAGS = -lineinfo -Xcompiler -pg

# Source files (only CUDA files that need nvcc)
CUDA_SOURCES = $(SRC_DIR)/main.cu \
               $(SRC_DIR)/ray_kernels.cu \
               $(SRC_DIR)/po_kernels.cu \
               $(SRC_DIR)/world_setup.cu \
               $(SRC_DIR)/cuda_utils.cu \
               $(SRC_DIR)/hitable_list.cu \
               $(SRC_DIR)/sphere.cu

# C++ source files (pure C++, no CUDA)
CPP_SOURCES = $(SRC_DIR)/config_parser.cpp \
              $(SRC_DIR)/printing_utils.cpp

# Object files (all in build directory)
CUDA_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SOURCES))
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))

# All objects
OBJECTS = $(CUDA_OBJECTS) $(CPP_OBJECTS)

# Default target (optimized)
all: NVCC_FLAGS += $(OPT_FLAGS)
all: CXX_FLAGS += -O3
all: $(TARGET)

# Debug build
debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: CXX_FLAGS += -g -O0 -DDEBUG
debug: $(TARGET)

# Profile build
profile: NVCC_FLAGS += $(PROFILE_FLAGS) $(OPT_FLAGS)
profile: CXX_FLAGS += -pg -O3
profile: $(TARGET)

# Create necessary directories (as a separate rule that runs first)
$(BUILD_DIR) $(BIN_DIR):
	@mkdir -p $@

# Link - ensure output goes to bin directory
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
	@echo "Build complete: $(TARGET)"

# Compile CUDA source files - ensure output goes to build directory
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

# Compile C++ source files - ensure output goes to build directory
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Clean - removes object files and executable
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BIN_DIR)
	@echo "Clean complete."

# Deep clean - also remove any output files
distclean: clean
	@rm -f *.csv *.dat *.log
	@rm -f output/rcs_results.csv
	@echo "Distribution clean complete."

# Run the program
run: all
	./$(TARGET)

# Run with specific frequency (example: make run-freq FREQ=5e9)
run-freq: all
	./$(TARGET) $(FREQ)

# Show what files would be built
show:
	@echo "Source files:"
	@echo "  CUDA: $(CUDA_SOURCES)"
	@echo "  C++:  $(CPP_SOURCES)"
	@echo ""
	@echo "Object files (in $(BUILD_DIR)):"
	@echo "  CUDA: $(CUDA_OBJECTS)"
	@echo "  C++:  $(CPP_OBJECTS)"
	@echo ""
	@echo "Target executable (in $(BIN_DIR)):"
	@echo "  $(TARGET)"

# Help
help:
	@echo "Available targets:"
	@echo "  make           - Build optimized version (default)"
	@echo "  make debug     - Build with debug symbols and no optimization"
	@echo "  make profile   - Build with profiling information"
	@echo "  make clean     - Remove build artifacts (build/ and bin/)"
	@echo "  make distclean - Remove all generated files"
	@echo "  make run       - Build and run the program"
	@echo "  make run-freq FREQ=<value> - Build and run with specific frequency"
	@echo "  make show      - Show source and object files"
	@echo ""
	@echo "Directory structure:"
	@echo "  Object files: $(BUILD_DIR)/"
	@echo "  Executable:   $(BIN_DIR)/"
	@echo ""
	@echo "Examples:"
	@echo "  make debug"
	@echo "  make run-freq FREQ=10e9"
	@echo "  make profile && ./bin/SagittaSBR"

.PHONY: all debug profile clean distclean run run-freq show help