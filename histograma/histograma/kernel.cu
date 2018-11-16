
#include "kernel.h"


//amountForThread - defines how many numbers each thread checks
__global__ void kernelHistogramCalc(int* inputArr, int* tmpHistogramArr, int numberRange)
{
	int threadId = threadIdx.x;
	int startIndex = threadId * CUDA_AMOUNT_FOR_THREAD;
	int partialStartCell = threadId * numberRange;	// the start cell in the tmpHistogram array for each thread
	int numToSave;
	int i;

	for (i = startIndex; i < (startIndex + CUDA_AMOUNT_FOR_THREAD); i++)
	{
		numToSave = inputArr[i];	//get the number from the main array (A)
		tmpHistogramArr[partialStartCell + numToSave]++;	//increase the count of the number in the tmep histogram
	}
}

__global__ void reduceToFinalHistogramKernel(int* tmpHistogramArr, int* finalHistogram, int numberRange)
{
	int i;

	for (i = 0; i < blockDim.x; i++)
		finalHistogram[threadIdx.x] += tmpHistogramArr[(i*numberRange) + threadIdx.x];
	

	/*for (i = 0; i < numberRange; i++)
	{
		for (j = 0; j < blockDim.x; j++)
		{
			step = (j * numberRange);
			finalHistogram[i] += tmpHistogramArr[i + step];
		}

	}*/
}


//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = manageCudaCalc(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t manageCudaCalc(int* inputArr, int inputArrSize, int* cudaHistogram, int numberRange, const int numOfCudaThreads)
{
    //int *dev_a = 0;
	int* cudaInputArr = 0;
    //int *dev_b = 0;
	int* cudaTmpHisto = 0;
    //int *dev_c = 0;
	int* cudaFinalHisto = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        freeCudaAllocations(3, cudaInputArr, cudaTmpHisto, cudaFinalHisto) ;
		return cudaStatus;
    }

    // Allocate GPU buffers for the input array
    cudaStatus = cudaMalloc((void**)&cudaInputArr, inputArrSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        freeCudaAllocations(3, cudaInputArr, cudaTmpHisto, cudaFinalHisto) ;
		return cudaStatus;
    }

	// Allocate GPU buffers for the temporary histogram (the large one 100 * 256 size)
	cudaStatus = cudaMalloc((void**)&cudaTmpHisto, (numberRange * numOfCudaThreads * sizeof(int)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        freeCudaAllocations(3, cudaInputArr, cudaTmpHisto, cudaFinalHisto) ;
		return cudaStatus;
    }

	// Allocate GPU buffers for the final cuda histogram (256 size) 
    cudaStatus = cudaMalloc((void**)&cudaFinalHisto, numberRange * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        freeCudaAllocations(3, cudaInputArr, cudaTmpHisto, cudaFinalHisto) ;
		return cudaStatus;
    }

    // Copy inputArr from host(CPU) memory to GPU buffers.
    cudaStatus = cudaMemcpy(cudaInputArr, inputArr, inputArrSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        freeCudaAllocations(3, cudaInputArr, cudaTmpHisto, cudaFinalHisto) ;
		return cudaStatus;
    }

	//const int numOfBlocks = (numOfCudaThreads / 1000) + 1;
    // Launch a kernel histogram calculation on the GPU 
	kernelHistogramCalc <<<1, numOfCudaThreads >>> (cudaInputArr, cudaTmpHisto, numberRange);
	

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernelHistogramCalc launch failed: %s\n", cudaGetErrorString(cudaStatus));
        freeCudaAllocations(3, cudaInputArr, cudaTmpHisto, cudaFinalHisto) ;
		return cudaStatus;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernelHistogramCalc!\n", cudaStatus);
        freeCudaAllocations(3, cudaInputArr, cudaTmpHisto, cudaFinalHisto) ;
		return cudaStatus;
    }

	//reduce the temporary histogram to a final short one, with only 1 thread to avoid collisions
	reduceToFinalHistogramKernel <<<1, numberRange>>> (cudaTmpHisto, cudaFinalHisto, numberRange);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "reduceToFinalHistogramKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		freeCudaAllocations(3, cudaInputArr, cudaTmpHisto, cudaFinalHisto);
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching reduceToFinalHistogramKernel!\n", cudaStatus);
		freeCudaAllocations(3, cudaInputArr, cudaTmpHisto, cudaFinalHisto) ;
		return cudaStatus;
	}

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(cudaHistogram, cudaFinalHisto, numberRange * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        freeCudaAllocations(3, cudaInputArr, cudaTmpHisto, cudaFinalHisto) ;
		return cudaStatus;
    }
    
    return cudaStatus;
}

void freeCudaAllocations(int numberOfPtrsToFree, ...)
{
	va_list arg_ptr;
	va_start(arg_ptr, numberOfPtrsToFree);

	for (int i = 0; i < numberOfPtrsToFree; i++) 
		cudaFree(va_arg(arg_ptr, int*));

	va_end(arg_ptr);
}
