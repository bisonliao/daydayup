#ifndef _BISON_TDIGEST_H_INCLUDED_
#define _BISON_TDIGEST_H_INCLUDED_

/*
https://github.com/tdunning/t-digest
*/

#include "centroid.h"



#define MAX_CENTROID_NUM  (1000)




typedef struct tdigest
{
    /* data */
    centroid_handle_t * centroidList[MAX_CENTROID_NUM];
    uint32_t centroidNum;
    uint64_t eleTotalNum;
    double min;
    double max;
    
} tdigest_handle_t;

int tdigest_init(tdigest_handle_t * t, double min, double max);
int tdigest_update(tdigest_handle_t * t, double ele);
int tdigest_getPercentile(tdigest_handle_t * t, int tp, double * v);
int tdigest_print(tdigest_handle_t * t);


#endif