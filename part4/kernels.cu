#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat (float *h, float *g, int N) {
	// float diff, sum=0.0;
	int j = threadIdx.x + blockDim.x * blockIdx.x + 1;
	int i = threadIdx.y + blockDim.y * blockIdx.y + 1;
	if (i < N - 1 && j < N - 1) {
		g[i*N + j]= 0.25 * ( h[ i*N + (j-1) ]+  // left
				     h[ i*N     + (j+1) ]+  // right
				     h[ (i-1)*N + j     ]+  // top
				     h[ (i+1)*N + j     ]); // bottom
		// diff = g[i*N + j] - h[i*N + j];
		// sum += diff * diff;
	}
}

__global__ void gpu_Diff(float *h, float *g, int N) {
	float diff;
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	if (i == 0 || i == N - 1) {
		if (j < N) h[i*N + j] = 0;
	} else if (i < N) {
		if (j == 0 || j == N - 1) {
			h[i*N + j] = 0;
		} else if (j < N) {
			diff = g[i*N + j] - h[i*N + j];
			h[i*N + j] = diff * diff;
		}
	}
}

__global__ void gpu_Reduce(float *g) {
  __shared__ int sdata[256];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = g[i];

    __syncthreads();
    // do reduction in shared mem
    for (int s=1; s < blockDim.x; s *=2)
    {
        int index = 2 * s * threadIdx.x;;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (threadIdx.x == 0)
        atomicAdd(g,sdata[0]);
}

