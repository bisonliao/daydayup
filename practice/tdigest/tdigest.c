#include "tdigest.h"
#include <time.h>

int tdigest_init(tdigest_handle_t * t, double min, double max)
{
    if (t == NULL) {return -1;}
    memset(t->centroidList, 0, sizeof(t->centroidList));
    t->centroidNum = 0;
    t->eleTotalNum = 0;

    t->min = min;
    t->max = max;
    double step = (max - min ) / INIT_CENTROID_NUM;

    int i;
    for (i = 0; i < INIT_CENTROID_NUM; ++i)
    {
        t->centroidList[i] = (centroid_handle_t*) malloc(sizeof(centroid_handle_t));
        if (t->centroidList[i] == NULL)
        {
            fprintf(stderr, "%s %d:failed to malloc\n", __FILE__, __LINE__);
            return -2;
        }
        centroid_init(t->centroidList[i]);
     
        if (i < (INIT_CENTROID_NUM *0.1)  )//first 100 centroids 
        {
            t->centroidList[i]->centroidCenter = min + step / 10 * i;
        }
        else if (i > (INIT_CENTROID_NUM*0.9)) // last 100 centroids
        {
            t->centroidList[i]->centroidCenter = max - step / 10 * (INIT_CENTROID_NUM-i);
        }
        else 
        {
            t->centroidList[i]->centroidCenter = min+i*step;
        }
        
        t->centroidNum += 1;
    }
    return 0;
}

static int tdigest_getCluster(tdigest_handle_t * t, double ele)
{
    // search by bruce force, it costs too much cpu from gprof analysis
    // maybe we can optimize it by KD tree or Ball Tree
    if (t->centroidNum < 1) { return -1;}
    int i;
    double oldDistance = doubleAbs(t->centroidList[0]->centroidCenter - ele);
    int oldIndex = 0;

    for (i = 1; i < t->centroidNum; ++i)
    {
        double distance = doubleAbs(t->centroidList[i]->centroidCenter - ele);
        if ( distance < oldDistance)
        {
            oldDistance = distance;
            oldIndex = i;
            continue;
        }
        else
        {
            break;
        }
    }
    return oldIndex;

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

    // if the centroid centers are NOT sequen, swap to keep them ordered
    int i;
    int swapflag = 0;
    
    i = centroidIndex;
    

    while (i+1 < t->centroidNum && t->centroidList[i]->centroidCenter > t->centroidList[i+1]->centroidCenter)
    {
        centroid_handle_t * a = t->centroidList[i+1];
        t->centroidList[i+1] = t->centroidList[i];
        t->centroidList[i] = a;
        i++;
        swapflag = 1;
        //fprintf(stdout, "%s %d:swap\n", __FILE__, __LINE__);
    }
    if (swapflag) { centroidIndex = i;}

    i = centroidIndex;
    swapflag = 0;

    while (i-1 >= 0 && t->centroidList[i]->centroidCenter < t->centroidList[i-1]->centroidCenter)
    {
        centroid_handle_t * a = t->centroidList[i-1];
        t->centroidList[i-1] = t->centroidList[i];
        t->centroidList[i] = a;
        i--;
        swapflag = 1;
        //fprintf(stdout, "%s %d:swap\n", __FILE__, __LINE__);
    }
    if (swapflag) { centroidIndex = i;}
#if 1
    // if the centroid has too many elements, then split it and insert into the list;
    if (t->centroidList[centroidIndex]->eleNum > 10000 &&
        t->centroidList[centroidIndex]->eleNum > (t->eleTotalNum / t->centroidNum * 5) &&
        t->centroidNum < MAX_CENTROID_NUM)
    {
        centroid_handle_t *a = t->centroidList[centroidIndex];

        centroid_handle_t *b = (centroid_handle_t*)malloc(sizeof(centroid_handle_t));
        if (b == NULL) {fprintf(stderr, "%s %d: malloc failed!\n", __FILE__, __LINE__); return -2;}
        centroid_init(b);

        centroid_handle_t *c = (centroid_handle_t*)malloc(sizeof(centroid_handle_t));
        if (c == NULL) {fprintf(stderr, "%s %d: malloc failed!\n", __FILE__, __LINE__); free(b); return -2;}
        centroid_init(c);


        ret = centroid_split(a, b, c);
        if (ret != 0)
        {
            fprintf(stderr, "%s %d: centroid_split failed!\n", __FILE__, __LINE__); 
            free(b);
            free(c);
            return -2;
        }

        if (t->centroidNum > (centroidIndex+1))
        {
            memmove( &t->centroidList[centroidIndex+2], 
                &t->centroidList[centroidIndex+1], 
                (t->centroidNum - (centroidIndex+1)) * sizeof(t->centroidList[0]) );
        }
        t->centroidList[centroidIndex] = b;
        t->centroidList[centroidIndex+1] = c;
        t->centroidNum++;
        free(a);
    }
#endif
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
    result += ((double)(cnt-sum))/t->centroidList[i]->eleNum * 2 * (t->centroidList[i]->max - t->centroidList[i]->min);
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
    // wo cao, tdigest implemented by bison is much slower than qsort.
    // gprof shows:clustering costs too much cpu.
    /*
                Call graph (explanation follows)


granularity: each sample hit covers 2 byte(s) for 0.00% of 243.17 seconds

index % time    self  children    called     name
                                                 <spontaneous>
[1]    100.0    0.98  242.18                 main [1]
                2.33  239.83 100000000/100000000     tdigest_update [2]
                0.02    0.01       1/1           tdigest_init [7]
                0.00    0.00       4/4           tdigest_getPercentile [9]
                0.00    0.00       1/1           tdigest_free [10]
-----------------------------------------------
                2.33  239.83 100000000/100000000     main [1]
[2]     99.6    2.33  239.83 100000000         tdigest_update [2]
              144.49   79.39 100000000/100000000     tdigest_getCluster [3]
                1.11   14.76 100000000/100000000     centroid_addElement [5]
                0.09    0.00      39/39          centroid_split [6]
                0.00    0.00      78/1078        centroid_init [8]
-----------------------------------------------
              144.49   79.39 100000000/100000000     tdigest_update [2]
[3]     92.1  144.49   79.39 100000000         tdigest_getCluster [3]
               79.39    0.00 537906649/637906649     doubleAbs [4]
-----------------------------------------------
               14.76    0.00 100000000/637906649     centroid_addElement [5]
               79.39    0.00 537906649/637906649     tdigest_getCluster [3]
[4]     38.7   94.14    0.00 637906649         doubleAbs [4]
*/
    srandom(time(NULL));
    if (argc < 2)
    {
        printf("use tdigest...\n");
        tdigest_handle_t handle;
        tdigest_init(&handle, 0, 10000);
        for (int i = 0; i < 100000000; ++i)
        {
            double ele = ((double)random()) / RAND_MAX * 10000;
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
    double *arr = malloc(100000000*sizeof(double));
    for (int i = 0; i < 100000000; ++i)
    {
        double ele = ((double)random()) / RAND_MAX * 10000;
        arr[i] = ele;
    }
    qsort(arr, 100000000, sizeof(double), compareDouble);
    printf("%f, %f, %f, %f\n", arr[5000000], arr[50000000], arr[90000000], arr[95000000]);

}