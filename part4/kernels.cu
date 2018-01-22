#include <math.h>
#include <float.h>
#include <cuda.h>

#define BLOCK_SIZE 256

__global__ void gpu_Reduce (float *s, int N, int skip) {
	__shared__ float sdata[BLOCK_SIZE];
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * skip;
	sdata[threadIdx.x] = i < N? s[i] : 0.0;
	__syncthreads();
	// Do reduction in shared mem
	for (int s=1; s < blockDim.x; s *=2) {
		int index = 2 * s * threadIdx.x;
		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) s[i] = sdata[0];
}

__global__ void gpu_Heat (float *h, float *g, int N) {
	int j = threadIdx.x + blockDim.x * blockIdx.x + 1;
	int i = threadIdx.y + blockDim.y * blockIdx.y + 1;
	if (i < N - 1 && j < N - 1) {
		g[i*N + j]= 0.25 * ( h[ i*N + (j-1) ]+  // left
				     h[ i*N     + (j+1) ]+  // right
				     h[ (i-1)*N + j     ]+  // top
				     h[ (i+1)*N + j     ]); // bottom
	}
}


__device__ void gpu_Diff_Reduce_Aux(float *h, float *g, float *sdata, int N) {
	float diff;
	int i_lin = blockIdx.x * blockDim.x + threadIdx.x;
	int i = i_lin / N;
	int j = i_lin - i * N;
	// Compute diff
	if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
		diff = g[i*N + j] - h[i*N + j];
		sdata[threadIdx.x] = diff * diff;
	} else {
		sdata[threadIdx.x] = 0.0; 
	}
	__syncthreads();
	// Do reduction in shared mem
	for (int s=1; s < blockDim.x; s *=2) {
		int index = 2 * s * threadIdx.x;;
		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}
}

__global__ void gpu_Diff_Reduce(float *h, float *g, float *s, int N) {
	__shared__ float sdata[BLOCK_SIZE];
	gpu_Diff_Reduce_Aux(h, g, sdata, N);
	// Write result for this block to global mem
	if (threadIdx.x == 0) s[blockIdx.x] = sdata[0];
}

__global__ void gpu_Diff_Reduce_Atomic(float *h, float *g, float *s, int N) {
	__shared__ float sdata[BLOCK_SIZE];
	gpu_Diff_Reduce_Aux(h, g, sdata, N);
	// Write result for this block to global mem
	if (threadIdx.x == 0) atomicAdd(s, sdata[0]);
}
