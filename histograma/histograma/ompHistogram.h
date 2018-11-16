#ifndef __OMPHIST_H
#define __OMPHIST_H

#include <stdlib.h>
#include <omp.h>

void ompHistogramaCalc(const int*const& arr, int size, int*& tmpHistogram, int numberRange);

void reduceToFinalOmpHistogram(const int*const& tmpOmpHisto, int*& finalOmpHist, int numberRange);


#endif // __OMPHIST_H
