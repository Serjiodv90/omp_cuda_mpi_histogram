#define _CRT_SECURE_NO_WARNINGS

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


#define ARRAY_SIZE 20000
#define NUMBER_RANGE 256
#define NUM_OF_PROCS 2
#define CUDA_NUM_THREADS 500
#define MASTER 0 
#define SLAVE 1

void checkAllocation(const void *ptr)
{
	if (!ptr)
	{
		printf("Allocation not good!");
		MPI_Finalize();
		exit(1);
	}
}

void initArray(int* a)
{
	srand(NULL);

	for (int i = 0; i < ARRAY_SIZE; i++)
		a[i] = rand() % NUMBER_RANGE;
	
}

void allocateArrays(int *inputArr, int* histogram, int* ompTmpHistogram, int* cudaTmpHistogram)
{
	inputArr = (int*)malloc(sizeof(int) * ARRAY_SIZE);
	checkAllocation(inputArr);
	histogram = (int*)calloc(NUMBER_RANGE, sizeof(int));
	checkAllocation(histogram);
	ompTmpHistogram = (int*)calloc(omp_get_num_threads() * NUMBER_RANGE, sizeof(int));
	checkAllocation(ompTmpHistogram);
	cudaTmpHistogram = (int*)calloc(CUDA_NUM_THREADS * NUMBER_RANGE, sizeof(int));
	checkAllocation(cudaTmpHistogram);
}

void printArray(const int* arr, int size)
{
	for (int i = 0; i < size; i++)
		printf(arr[i] + " ");
	printf("\n");
	
}

//void freeArrayAllocation(int* arrays...)
//{
//	
//}

/* arr - the array to check the histograma
 * size - the pratiall size of the array for the omp threads
*/
void ompHistogramaCalc(int* arr, int size, int* tmpHistogram)
{
	int nThreads = omp_get_num_threads();
	tmpHistogram = (int*)calloc((nThreads * NUMBER_RANGE), sizeof(int));		//init with 0
	checkAllocation(tmpHistogram);

#pragma omp parallel for 
	for (int i = 0; i < size; i++)
	{
		int threadId = omp_get_thread_num();
		int partalStartCell = threadId * NUMBER_RANGE;	// the start cell in the tmpHistogram array for each thread
		int numToSave = arr[i];	//get the number from the main array (A)
		tmpHistogram[numToSave + partalStartCell]++;	//increase the count of the number in the partial histogram
	}
}





void main(int argc, char *argv[])
{
	int myId, numprocs;
	MPI_Status status;
	int *inputArr, *histogram, *procInputArray, *ompTmpHistogram, *cudaTmpHistogram;

	allocateArrays(inputArr, histogram, ompTmpHistogram, cudaTmpHistogram);	     

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	
	if (numprocs != NUM_OF_PROCS)
	{
		printf("Should be only 2 processes\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (myId == MASTER)
	{
		initArray(inputArr);
		printf("Input array after init: \n");
		printArray(inputArr, ARRAY_SIZE);
		fflush(stdout);

		MPI_Send((inputArr + ARRAY_SIZE / 2), ARRAY_SIZE / 2, MPI_INT, SLAVE, 0, MPI_COMM_WORLD);
	}
	else
	{
		procInputArray = (int*)malloc(sizeof(int) * (ARRAY_SIZE / 2));
		checkAllocation(procInputArray);
		MPI_Recv(procInputArray, (ARRAY_SIZE / 2), MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
	}

	ompHistogramaCalc(procInputArray, ARRAY_SIZE / 2, ompTmpHistogram);


	MPI_Finalize();




}
