#ifndef __KERNEL_H
#define __KERNEL_H




#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdarg.h>
#include "mainHelper.h"

__global__ void kernelHistogramCalc(int* inputArr, int* tmpHistogramArr, int numberRange);
__global__ void reduceToFinalHistogramKernel(int* tmpHistogramArr, int* finalHistogram, int numberRange);

cudaError_t manageCudaCalc(int* inputArr, int inputArrSize, int* cudaHistogram, int numberRange, int numOfCudaThreads);
void freeCudaAllocations(int numberOfPtrsToFree, ...);





#endif // !__KERNEL_H