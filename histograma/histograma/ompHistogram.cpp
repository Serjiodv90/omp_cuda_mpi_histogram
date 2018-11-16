#include "ompHistogram.h"



/* arr - the array to check the histograma
* size - the pratiall size of the array for the omp threads
*/
void ompHistogramaCalc(const int*const& arr, int size, int*& finalOmpHistogram, int numberRange)
{
	int nThreads = omp_get_max_threads();
	int* tmpHistogram = (int*)calloc((nThreads * numberRange), sizeof(int));

#pragma omp parallel for 
	for (int i = 0; i < size; i++)
	{
		int threadId = omp_get_thread_num();
		int partialStartCell = threadId * numberRange;	// the start cell in the tmpHistogram array for each thread
		int numToSave = arr[i];	//get the number from the main array (A)
		tmpHistogram[partialStartCell + numToSave]++;	//increase the count of the number in the tmep histogram
	}

	reduceToFinalOmpHistogram(tmpHistogram, finalOmpHistogram, numberRange);
}

void reduceToFinalOmpHistogram(const int*const& tmpOmpHisto, int*& finalOmpHist, int numberRange)
{
	int ompNumOfThreads = omp_get_max_threads();
	
	for (int i = 0; i < numberRange; i++)
	{
		for (int j = 0; j < ompNumOfThreads; j++)
		{
			int step = (j * numberRange);
			finalOmpHist[i] += tmpOmpHisto[i + step];
		}

	}
}