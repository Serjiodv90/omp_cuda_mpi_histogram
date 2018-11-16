/*	AUTHORS:
*	Hadar Pur 308248533
*	Sergei Dvorjin 316859552
*/

#include "mainHelper.h"



void main(int argc, char *argv[])
{
	int myId, numprocs;
	MPI_Status status;
	int *inputArr, *histogram, *ompTmpHistogram, *tmpHistogramFromSlave, *cudaTmpHistogram;


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	if (numprocs != NUM_OF_PROCS)
	{
		printf("Should be only 2 processes\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	allocateArrays(inputArr, histogram, ompTmpHistogram, cudaTmpHistogram);

	int arraySizeForEachProc = ARRAY_SIZE / numprocs;	//the array size that each proccess get

	if (myId == MASTER)
	{
		initArray(inputArr);

		printf("The sequencial histogram: \n");
		printArray(sequencialHistogram(inputArr, ARRAY_SIZE, NUMBER_RANGE), NUMBER_RANGE);
		fflush(stdout);

		MPI_Send((inputArr + arraySizeForEachProc), arraySizeForEachProc, MPI_INT, SLAVE, 0, MPI_COMM_WORLD);

	}
	else
	{
		MPI_Recv(inputArr, arraySizeForEachProc, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
	}

	int arraySizeForOmpAndCuda = arraySizeForEachProc / 2;

	//calculate histogram with omp for each process
	ompHistogramaCalc(inputArr, arraySizeForOmpAndCuda, ompTmpHistogram, NUMBER_RANGE);

	//calculate histogram with cuda for each process. (inputArr + arraySizeForOmpAndCuda) - means that cuda get the second half of the input in the process
	cudaError_t cudaStatus = manageCudaCalc((inputArr + arraySizeForOmpAndCuda), arraySizeForOmpAndCuda, cudaTmpHistogram, NUMBER_RANGE, cudaNumOfThreads());

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "manageCudaCalc failed!");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	//combine both cuda and omp temporar histograms to the final histogram of each proccess
	combineTwoHistograms(ompTmpHistogram, cudaTmpHistogram, histogram);

	if (myId == SLAVE)
	{
		MPI_Send(histogram, NUMBER_RANGE, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
		freeArrayAllocation(4, inputArr, histogram, ompTmpHistogram, cudaTmpHistogram);
	}
	else
	{
		tmpHistogramFromSlave = (int*)calloc(NUMBER_RANGE, sizeof(int));
		MPI_Recv(tmpHistogramFromSlave, NUMBER_RANGE, MPI_INT, SLAVE, 0, MPI_COMM_WORLD, &status);

		combineTwoHistograms(tmpHistogramFromSlave, histogram, histogram);
		int isTheHistogramCorrect = isHistogramsEqual(inputArr, histogram, NUMBER_RANGE, ARRAY_SIZE);
		printf("\nproccess id: %d\n", myId);
		printf("are the arrays equal: %s\n", isTheHistogramCorrect == 1 ? "true!" : "false");
		printf("\nThe combined histogram: \n");
		printArray(histogram, NUMBER_RANGE);
		fflush(stdout);

		freeArrayAllocation(4, inputArr, histogram, ompTmpHistogram, cudaTmpHistogram);

	}


	MPI_Finalize();

}
