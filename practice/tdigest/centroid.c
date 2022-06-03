#include "centroid.h"

int centroid_init(centroid_handle_t * c)
{
    if (c == NULL) { return -1;}
    c->centroidCenter = 0;
    c->eleNum = 0;
    
    
    return 0;
}

int centroid_addElement(centroid_handle_t * c, double ele)
{
    if (c == NULL) { return -1;}
    double r = doubleAbs(ele - c->centroidCenter);
    c->centroidCenter = (c->centroidCenter * c->eleNum + ele) / (c->eleNum + 1);
    c->eleNum += 1;

    if (c->eleNum == 1)
    {
        c->min = ele;
        c->max = ele;
    }
    else
    {
        if (ele < c->min) {c->min = ele;}
        if (ele > c->max) {c->max = ele;}
    }


    
    return 0;
}

int centroid_split(centroid_handle_t * a, centroid_handle_t * b, centroid_handle_t * c)
{
    if (a == NULL || b == NULL || c == NULL) { return -1;}

    b->centroidCenter = a->centroidCenter - (a->centroidCenter - a->min) / 2;
    c->centroidCenter = a->centroidCenter + (a->max - a->centroidCenter)/2;

    b->eleNum = a->eleNum / 2;
    c->eleNum = a->eleNum - b->eleNum;

    b->min = a->min;
    b->max = a->centroidCenter;
    c->min = a->centroidCenter;
    c->max = a->max;

  

    return 0;

}
double doubleAbs(double d)
{
    if (d < 0) return -d;
    return d;
}

