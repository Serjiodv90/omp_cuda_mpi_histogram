#ifndef __MAINHELPER_H
#define __MAINHELPER_H

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ompHistogram.h"
#include "kernel.h"


#define _CRT_SECURE_NO_WARNINGS


#define ARRAY_SIZE 200000
#define NUMBER_RANGE 256
#define NUM_OF_PROCS 2
#define MASTER 0 
#define SLAVE 1
#define CUDA_AMOUNT_FOR_THREAD 500

#define ompNumOfThreads()	(omp_get_max_threads())
#define ompTempHistogramArraySize() (omp_get_max_threads() * NUMBER_RANGE)
#define cudaNumOfThreads() (((ARRAY_SIZE) / (NUM_OF_PROCS * 2)) / CUDA_AMOUNT_FOR_THREAD)	//every cuda thread deals with up to 500 numbers of the input = 100 cuda threads


void checkAllocation(const void *ptr);

void allocateArrays(int*& inputArr, int*& histogram, int*& ompTmpHistogram, int*& cudaTmpHistogram);

void printArray(const int*const& arr, int size);

void initArray(int* a);

void freeArrayAllocation(int numberOfPtrsToFree, ...);


int* sequencialHistogram(const int*const& arr, int arraySize, int numberRange);

bool isHistogramsEqual(const int*const& arrForHistogram, const int*const& histogramToCheck, int numberRange, int arraySize);

void combineOmpHistograms(const int*const& arr1, const int*const& arr2, int*&resulatHistogram);

void combineTwoHistograms(const int*const& ompHisto, const int*const& cudaHisto, int*& finalHisto);









#endif // !__MAINHELPER_H


