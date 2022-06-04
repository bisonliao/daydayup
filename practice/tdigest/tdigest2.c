#include "tdigest2.h"
#include <time.h>

#include "data.c"

int tdigest_init(tdigest_handle_t * t, double min, double max)
{
    if (t == NULL) {return -1;}
    memset(t->centroidList, 0, sizeof(t->centroidList));
    t->centroidNum = 0;
    t->eleTotalNum = 0;

    t->min = min;
    t->max = max;
    double step = (max - min ) / MAX_CENTROID_NUM;

    int i;
    for (i = 0; i < MAX_CENTROID_NUM; ++i)
    {
        t->centroidList[i] = (centroid_handle_t*) malloc(sizeof(centroid_handle_t));
        if (t->centroidList[i] == NULL)
        {
            fprintf(stderr, "%s %d:failed to malloc\n", __FILE__, __LINE__);
            return -2;
        }
        centroid_init(t->centroidList[i]);
        t->centroidList[i]->centroidCenter = min+i*step;
        t->centroidNum += 1;
    }
    return 0;
}

static int tdigest_getCluster(tdigest_handle_t * t, double ele)
{
    double step = (t->max - t->min ) / MAX_CENTROID_NUM;
    
    int index =  (ele - t->min) / step;
    if (index >= t->centroidNum)
    {
        return t->centroidNum-1;
    }

}


int tdigest_update(tdigest_handle_t * t, double ele)
{
    int ret;
    if (t == NULL) {return -1;}
    int centroidIndex = tdigest_getCluster(t, ele);
    ret = centroid_addElement(t->centroidList[centroidIndex], ele);
    if (ret != 0)
    {
        fprintf(stderr, "%s %d: centroid_addElement failed!\n", __FILE__, __LINE__); 
        return -2;
    }
    t->eleTotalNum++;

   
    return 0;
        
}
int tdigest_getPercentile(tdigest_handle_t * t, int tp, double * v)
{
    if (t == NULL || v == NULL ) { return -1;}
    uint64_t cnt = (double)t->eleTotalNum * tp / 100;
    uint64_t sum = 0;
    int i;
    for (i = 0; i < t->centroidNum; ++i)
    {
        if (sum + t->centroidList[i]->eleNum  > cnt)
        {
            break;
        }
        sum += t->centroidList[i]->eleNum;
    }
    if (i >= t->centroidNum)
    {
        return -1;
    }
    
    double result =  t->centroidList[i]->min;
    result += ((double)(cnt-sum))/t->centroidList[i]->eleNum  * (t->centroidList[i]->max - t->centroidList[i]->min);
    *v = result;
    return 0;

}
int tdigest_print(tdigest_handle_t * t)
{
    fprintf(stdout, "centroid num:%u, elements num:%lu\n", t->centroidNum, t->eleTotalNum);
    fprintf(stdout, "-- eleNum center  radius--\n");
    for (int i = 0; i < t->centroidNum; i++)
    {
        fprintf(stdout, "#%d %u %f %f\n", i, t->centroidList[i]->eleNum, t->centroidList[i]->centroidCenter, t->centroidList[i]->min);
    }
    fprintf(stdout, "\n");
    return 0;
}
int tdigest_free(tdigest_handle_t * t)
{
    for (int i = 0; i < t->centroidNum; i++)
    {
        free(t->centroidList[i]);
        t->centroidList[i] = NULL;
    }
    t->centroidNum = 0;
    t->eleTotalNum = 0;

    t->min = 0;
    t->max = 0;
}
int compareDouble(const void * a, const void * b)
{
    double aa = *(double*)a;
    double bb = *(double*)b;
    if (aa < bb) return -1;
    if (aa == bb) return 0;
    if (aa > bb) return 1;
}

int main(int argc, char ** argv)
{

    #define ELE_NUM (1000000)
  
    srandom(time(NULL));
    if (argc < 2)
    {
        printf("use tdigest...\n");
        tdigest_handle_t handle;
        tdigest_init(&handle, 0, 10000);
        for (int i = 0; i < ELE_NUM; ++i)
        {
            //double ele = ((double)random()) / RAND_MAX * 10000;
            double ele = data[i];
            tdigest_update(&handle, ele);
            
        }
        //tdigest_print(&handle);

        double v;
        
        tdigest_getPercentile(&handle, 5, &v);printf("%f\n", v);
        tdigest_getPercentile(&handle, 50, &v);printf("%f\n", v);
        tdigest_getPercentile(&handle, 90, &v);printf("%f\n", v);
        tdigest_getPercentile(&handle, 95, &v);printf("%f\n", v);

        tdigest_free(&handle);
        return 0;
    }
    printf("use qsort...\n");
    double *arr = malloc(ELE_NUM*sizeof(double));
    for (int i = 0; i < ELE_NUM; ++i)
    {
        //double ele = ((double)random()) / RAND_MAX * 10000;
        double ele = data[i];
        arr[i] = ele;
    }
    qsort(arr, ELE_NUM, sizeof(double), compareDouble);
    printf("%f, %f, %f, %f\n", arr[ELE_NUM*5/100], arr[ELE_NUM*50/100], arr[ELE_NUM*9/10], arr[ELE_NUM*95/100]);

}