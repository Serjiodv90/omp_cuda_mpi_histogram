
#include "mainHelper.h"

void checkAllocation(const void *ptr)
{
	if (!ptr)
	{
		printf("Allocation not good!");
		MPI_Finalize();
		exit(1);
	}
}



void allocateArrays(int*& inputArr, int*& histogram, int*& ompTmpHistogram, int*& cudaTmpHistogram)
{
	inputArr = (int*)malloc(sizeof(int) * ARRAY_SIZE);
	checkAllocation(inputArr);
	histogram = (int*)calloc(NUMBER_RANGE, sizeof(int));
	checkAllocation(histogram);
	ompTmpHistogram = (int*)calloc(/*ompTempHistogramArraySize()*/NUMBER_RANGE, sizeof(int));
	checkAllocation(ompTmpHistogram);
	cudaTmpHistogram = (int*)calloc(/*cudaNumOfThreads() **/ NUMBER_RANGE, sizeof(int));
	checkAllocation(cudaTmpHistogram);
}

void printArray(const int*const& arr, int size)
{
	for (int i = 0; i < size; i++)
		printf("%d ,", arr[i]);
	printf("\n");

}

void initArray(int* a)
{
	srand(time(NULL));

	for (int i = 0; i < ARRAY_SIZE; i++)
		a[i] = rand() % NUMBER_RANGE;
}

void freeArrayAllocation(int numberOfPtrsToFree, ...)
{
	va_list arg_ptr;
	va_start(arg_ptr, numberOfPtrsToFree);

	for (int i = 0; i < numberOfPtrsToFree; i++)
		free(va_arg(arg_ptr, int*));

	va_end(arg_ptr);
}



/*
*	create the histogram from arr, and returns the reference to it
*/
int* sequencialHistogram(const int*const& arr, int arraySize, int numberRange)
{
	int* tmpArr = (int*)calloc(numberRange, sizeof(int));

	for (int i = 0; i < arraySize; i++)
		tmpArr[arr[i]]++;

	return tmpArr;
}

/*
*	creates a sequencial histogram from arrForHistogram, and compares it to the histogramToCheck
*/
bool isHistogramsEqual(const int*const& arrForHistogram, const int*const& histogramToCheck, int numberRange, int arraySize)
{
	int* seqHistogram = sequencialHistogram(arrForHistogram, arraySize, numberRange);

	for (int i = 0; i < numberRange; i++)
	{
		if (seqHistogram[i] != histogramToCheck[i])
			return 0;
	}

	return 1;
}

//combine 2 omp histograms from 2 procceses
void combineOmpHistograms(const int*const& arr1, const int*const& arr2, int*&resulatHistogram)
{
	//	int ompNumOfThreads = omp_get_max_threads();
	for (int i = 0; i < NUMBER_RANGE; i++)
	{
		for (int j = 0; j < ompNumOfThreads(); j++)
		{
			int step = (j * NUMBER_RANGE);
			resulatHistogram[i] += arr1[i + step] + arr2[i + step];
		}

	}
}

void combineTwoHistograms(const int*const& histo1, const int*const& histo2, int*& finalHisto)
{
	for (int i = 0; i < NUMBER_RANGE; i++)
		finalHisto[i] = histo1[i] + histo2[i];
}
