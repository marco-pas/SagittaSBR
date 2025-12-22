CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# Debug vs Release
NVCC_DBG       = -g -G
#NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64
GENCODE_FLAGS  = -gencode arch=compute_80,code=sm_80

SRCS = main.cu
INCS = vec3.h ray.h hitable.h hitable_list.h sphere.h box.h

TARGET = SBR
OBJ    = SBR.o

# Build executable
$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $(TARGET) $(OBJ)

# Compile object
$(OBJ): $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -c main.cu -o $(OBJ)

# Run targets
run: $(TARGET)
	./$(TARGET)

profile_basic: $(TARGET)
	nvprof ./$(TARGET)

profile_metrics: $(TARGET)
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./$(TARGET)

# Cleanup
clean:
	rm -f $(TARGET) $(OBJ)
