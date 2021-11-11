#ifndef SSTable_h_included
#define SSTable_h_included

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>
#include <arpa/inet.h>
#include "../skiplist/skiplist.h"


extern skiplist_cmpfunc_t sstable_score_cmp;



typedef struct 
{
    FILE * idx_fp;
    FILE * dat_fp;
    int level;
    int subscript;
    mode_t mode;
    int indexCnt;
    skiplist_buffer_t minScore;
    skiplist_buffer_t maxScore;
    unsigned char * idx_buf;
    unsigned char * dat_buf;
} ssfile_ctx;


int ssfile_init(ssfile_ctx * ctx, int level, int subscript, mode_t mode);
int ssfile_init2(ssfile_ctx* ctx, int level, const char * filename, mode_t mode);
int ssfile_append(ssfile_ctx * ctx, const skiplist_buffer_t * score, const skiplist_buffer_t * data);
int ssfile_read(ssfile_ctx * ctx, skiplist_buffer_t * score, skiplist_buffer_t * data);
int ssfile_final(ssfile_ctx * ctx, int sucFlag);

// merged file 
typedef struct 
{
    char filename[1024];
    skiplist_buffer_t beginScore;
    skiplist_buffer_t endScore;
} merged_file_t;



int sstable_getNextFileIndex(int level, int start);
int sstable_printIdxFile(const char  * filename);
void sstable_getfilename(char * filename, int level, int subscript, int isIndex, int isTmp);
void sstable_getpath(char * path, int level);
int sstable_sink();
int sstable_getAllFilesToMerge(int level, char filenames[][1024], int *maxNum);
int sstable_getAllMergedFiles(int level, merged_file_t * files, int *maxNum);
int sstable_getScoreRange(int level, merged_file_t * files, int cnt);
int fileMerge(const char * filename, int destLevel);

#define MAX_FILENUM_PER_LEVEL (1000)


#endif