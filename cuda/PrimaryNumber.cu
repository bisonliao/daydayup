
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory.h>
#include <stdlib.h>

// these numbers can not be larger, it is not because of GPU global memory size limit(for example:flag size),
// maybe it is limits by other level memory size.
const unsigned int blockNum = 1024;
const unsigned int threadNum = 1024;

__global__ void isDivisible(unsigned long value, unsigned long start, unsigned char *flag)
{
	//int threadid = threadIdx.x + blockNum * blockIdx.x;
	int threadid = blockDim.x * blockIdx.x + threadIdx.x;
	if ((value % (start + threadid)) == 0)
	{
		flag[threadid] = 1;
	}
}

int checkPrimary(unsigned long value)
{
	cudaError_t cudaStatus;
	unsigned char * dev_flag;
	unsigned char * host_flag;
	
	const  int size =  blockNum * threadNum;

	host_flag = (unsigned char*)malloc(size * sizeof(unsigned char));
	if (host_flag == NULL)
	{
		fprintf(stderr, "malloc failed!");
		return -1;
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_flag, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	for (;; value++)
	{
		cudaMemset(dev_flag, 0, size * sizeof(unsigned char));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemset failed!");
			goto Error;
		}
		bool isPrimary = true;
		for (unsigned long i = 2; i <= (value / 2); i += size)
		{
			isDivisible <<<blockNum, threadNum >>> (value, i, dev_flag);
			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "isDivisible launch failed: %s\n", cudaGetErrorString(cudaStatus));
				goto Error;
			}
			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching isDivisible:%s!\n", cudaStatus, cudaGetErrorString(cudaStatus));
				goto Error;
			}

			// Copy output vector from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(host_flag, dev_flag, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}
			
			for (int j = 0; j < size; ++j)
			{
				if (host_flag[j])
				{
					isPrimary = false;
					//printf("[%llu] [%llu]\n",value, i + j);
					break;
				}
			}
			if (isPrimary == false)
			{
				break;
			}

		}
		if (isPrimary)
		{
			printf("%llu is a primary number\n", value);
		}

	}
	


Error:

	cudaFree(dev_flag);
	free(host_flag);
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return -1;
	}
	return 0;

}
