#ifndef _BISON_CENTROID_H_INCLUDED_
#define _BISON_CENTROID_H_INCLUDED_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>


typedef struct 
{
    /* data */
    double centroidCenter;
    uint32_t eleNum;
    double min;
    double max;

} centroid_handle_t;

double doubleAbs(double d);

int centroid_init(centroid_handle_t * c);
int centroid_addElement(centroid_handle_t * c, double ele);
int centroid_split(centroid_handle_t * a, centroid_handle_t * b, centroid_handle_t * c);


#endif