# CUDA路径
CUDA_PATH = /usr/local/cuda

# PJ-lab A100
# MPI_PATH = /usr/lib/x86_64-linux-gnu/openmpi
# Ali-dsw
MPI_PATH = /usr/local/mpi

# 编译器
CXX = g++

# 编译选项
CXXFLAGS = -std=c++20 -O3 -I$(CUDA_PATH)/include -I/usr/include -I$(MPI_PATH)/include

# PJ-lab A100 : append -lmpi_cxx
LDFLAGS = -L$(CUDA_PATH)/lib64 -L$(MPI_PATH)/lib -lcudart -lnccl -lmpi

# 目标文件
TARGET = bwtest.out

# 源文件
SRCS = main.cc

# 生成目标
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(TARGET)