NVCC 	= /opt/cuda/8.0/bin/nvcc
CUDAPATH = /opt/cuda/8.0
# NVCC 	= /opt/cuda/4.1/bin/nvcc
# CUDAPATH = /opt/cuda/4.1
# NVCC = /opt/cuda/bin/nvcc
# CUDAPATH = /opt/cuda
# NVCC = /usr/local/cuda/bin/nvcc
# CUDAPATH = /usr/local/cuda

NVCCFLAGS = -I$(CUDAPATH)/include #--ptxas-options=-v 
LFLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm

# Reduction block size:
BLOCK_SIZE=256

# Modes: CPU, GPU_DIFF, GPU_REDUCE, GPU_ATOMIC
RESIDUAL=CPU

ALL	= heatCUDA
all: $(ALL)

kernels.o: kernels.cu 
	$(NVCC) -c -g $(NVCCFLAGS) $+ $(LFLAGS) -DBLOCK_SIZE=$(BLOCK_SIZE) -DRESIDUAL_$(RESIDUAL) -o $@ 

heatCUDA: heatCUDA.cu kernels.o
	$(NVCC) -g -O2 $(NVCCFLAGS) -DBLOCK_SIZE=$(BLOCK_SIZE) -DRESIDUAL_$(RESIDUAL) $+ $(LFLAGS) -o $@

clean:
	rm -fr $(ALL) *.o *.prv *.pcf *.row *.sym *.mpits set-0

