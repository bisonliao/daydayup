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
#include "skiplist.h"

// merged file 
typedef struct 
{
    char filename[1024];
    skiplist_buffer_t beginScore;
    skiplist_buffer_t endScore;
} merged_file_t;

extern skiplist_cmpfunc_t sstable_score_cmp;


int sstable_printIdxFile(const char  * filename);
void sstable_getfilename(char * filename, int level, int subscript, int isIndex, int isTmp);
void sstable_getpath(char * path, int level);
int sstable_sink();
int sstable_getAllFilesToMerge(int level, char filenames[][1024], int *maxNum);
int sstable_getAllMergedFiles(int level, merged_file_t * files, int *maxNum);
int sstable_getScoreRange(int level, merged_file_t * files, int cnt);

#define MAX_FILENUM_PER_LEVEL (1000)


#endif