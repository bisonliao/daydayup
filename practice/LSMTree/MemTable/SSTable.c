#include "SSTable.h"
#include <sys/types.h>
#include <dirent.h>

skiplist_cmpfunc_t sstable_score_cmp = skiplist_compare_score_default; 

int sstable_printIdxFile(const char  * filename)
{
    FILE * fp = NULL;
    if (filename == NULL)
    {
        return -1;
    }
    fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        perror(filename);
        return -1;
    }
    // range length
    fseek(fp, 0-(long)(sizeof(uint32_t)), SEEK_END);
    uint32_t rangeLen;
    if (fread(&rangeLen, sizeof(uint32_t), 1, fp) != 1)
    {
        perror("fread:");
        fclose(fp);
        fp = NULL;
        return -1;
    }
    rangeLen = ntohl(rangeLen);
    printf("range part length:%u\n", rangeLen);
    // min score
    fseek(fp, 0-(long)rangeLen, SEEK_END);
    uint32_t scoreLen;
    if (fread(&scoreLen, sizeof(uint32_t), 1, fp) != 1)
    {
        perror("fread:");
        fclose(fp);
        fp = NULL;
        return -1;
    }
    scoreLen = ntohl(scoreLen);
    printf("min score length:%u\n", scoreLen);
    if (scoreLen == sizeof(int))
    {
        int score;
        if (fread(&score, sizeof(int), 1, fp) != 1)
        {
            perror("fread:");
            fclose(fp);
            fp = NULL;
            return -1;
        }
        
        printf("min score :%d\n", score);
    }
    else
    {
        fseek(fp, scoreLen, SEEK_CUR);
    }
    // max score
    if (fread(&scoreLen, sizeof(uint32_t), 1, fp) != 1)
    {
        perror("fread:");
        fclose(fp);
        fp = NULL;
        return -1;
    }
    scoreLen = ntohl(scoreLen);
    printf("max score length:%u\n", scoreLen);
    if (scoreLen == sizeof(int))
    {
        int score;
        if (fread(&score, sizeof(int), 1, fp) != 1)
        {
            perror("fread:");
            fclose(fp);
            fp = NULL;
            return -1;
        }
        
        printf("max score :%d\n", score);
    }
    else
    {
        fseek(fp, scoreLen, SEEK_CUR);
    }
    // index
    uint32_t indexCnt;
    fseek(fp, 0, SEEK_SET);
    if (fread(&indexCnt, sizeof(indexCnt), 1, fp) != 1)
    {
        perror("fread:");
        fclose(fp);
        fp = NULL;
        return -1;
    }
    indexCnt = ntohl(indexCnt);
    uint32_t i;
    for (i = 0; i < indexCnt; ++i)
    {
        // score 
        if (fread(&scoreLen, sizeof(uint32_t), 1, fp) != 1)
        {
            perror("fread:");
            fclose(fp);
            fp = NULL;
            return -1;
        }
        scoreLen = ntohl(scoreLen);
        printf("%u\t", scoreLen);
        if (scoreLen == sizeof(int))
        {
            int score;
            if (fread(&score, sizeof(int), 1, fp) != 1)
            {
                perror("fread:");
                fclose(fp);
                fp = NULL;
                return -1;
            }
            
            printf("%d\t", score);
        }
        else
        {
            fseek(fp, scoreLen, SEEK_CUR);
        }
        // offset
        uint32_t offset;
        if (fread(&offset, sizeof(offset), 1, fp) != 1)
        {
            perror("fread:");
            fclose(fp);
            fp = NULL;
            return -1;
        }
        offset = ntohl(offset);
        printf("%u\n", offset);
    }
    long pos1 = ftell(fp);
    fseek(fp, 0, SEEK_END);
    long pos2 = ftell(fp);
    printf("%ld, %d\n", pos2 - pos1, rangeLen);

    fclose(fp);
    fp = NULL;
    return 0;

}


int sstable_getAllFilesToMerge(int level, char filenames[][1024], int *maxNum)
{
    char path[1024];
    int cnt = 0;
    
    sstable_getpath(path, level);
    DIR * dir = opendir(path);
    if (dir == NULL)
    {
        perror(path);
        return -1;
    }
    struct dirent * ent;
    while ( (ent = readdir(dir)) != NULL && cnt < *maxNum)
    {
        int d_len = strlen(ent->d_name);
        if ( strncmp(ent->d_name, "sstable_", 8) == 0 && d_len > 15 &&
        strcmp(ent->d_name+(d_len-4), ".idx") == 0)
        {
            strncpy(filenames[cnt++], ent->d_name, d_len-4); // subfix is ignore
        }
    }
    *maxNum = cnt;
    qsort(filenames, cnt, sizeof(filenames[0]), strcmp);

    return 0;
}
int sstable_getScoreRange(int level, merged_file_t * files, int cnt)
{
    int i;
    for (i = 0; i < cnt; ++i)
    {
        char filename[1024];
        char path[1024];
        sstable_getpath(path, level);
        sprintf(filename, "%s/%s.idx", path, files[i].filename);
        FILE * fp = NULL;
        fp = fopen(filename, "rb");
        if (fp == NULL)
        {
            perror(filename);
            return -1;
        }
        // range length
        fseek(fp, 0 - (long)(sizeof(uint32_t)), SEEK_END);
        uint32_t rangeLen;
        if (fread(&rangeLen, sizeof(uint32_t), 1, fp) != 1)
        {
            perror("fread:");
            fclose(fp);
            fp = NULL;
            return -1;
        }
        rangeLen = ntohl(rangeLen);
        //printf("range part length:%u\n", rangeLen);
        // min score
        fseek(fp, 0 - (long)rangeLen, SEEK_END);
        uint32_t scoreLen;
        if (fread(&scoreLen, sizeof(uint32_t), 1, fp) != 1)
        {
            perror("fread:");
            fclose(fp);
            fp = NULL;
            return -1;
        }
        scoreLen = ntohl(scoreLen);
        files[i].beginScore.len = scoreLen;
        files[i].beginScore.ptr = malloc(scoreLen);
        if (files[i].beginScore.ptr  == NULL)
        {
            fprintf(stderr, "malloc failed\n");
            fclose(fp);
            fp = NULL;
            return -1;
        }
        if (fread(files[i].beginScore.ptr, 1, scoreLen, fp) != scoreLen)
        {
            perror("fread");
            skiplist_free_buffer(&files[i].beginScore);
            skiplist_free_buffer(&files[i].endScore);
            fclose(fp);
            fp = NULL;
            return -1;
        }
         // max score
        if (fread(&scoreLen, sizeof(uint32_t), 1, fp) != 1)
        {
            perror("fread:");
            skiplist_free_buffer(&files[i].beginScore);
            skiplist_free_buffer(&files[i].endScore);
            fclose(fp);
            fp = NULL;
            return -1;
        }
        scoreLen = ntohl(scoreLen);
        files[i].endScore.len = scoreLen;
        files[i].endScore.ptr = malloc(scoreLen);
        if (files[i].endScore.ptr  == NULL)
        {
            fprintf(stderr, "malloc failed\n");
            skiplist_free_buffer(&files[i].beginScore);
            skiplist_free_buffer(&files[i].endScore);
            fclose(fp);
            fp = NULL;
            return -1;
        }
        if (fread(files[i].endScore.ptr, 1, scoreLen, fp) != scoreLen)
        {
            perror("fread");
            skiplist_free_buffer(&files[i].beginScore);
            skiplist_free_buffer(&files[i].endScore);
            fclose(fp);
            fp = NULL;
            return -1;
        }
        fclose(fp);
        fp = NULL;
        
    }
}
static int compare_two_merged_file(const void * a, const void *b)
{
    merged_file_t * aa = (merged_file_t * )a;
    merged_file_t * bb = (merged_file_t * )b;
    return sstable_score_cmp(&aa->beginScore, &bb->beginScore);
}
int sstable_getAllMergedFiles(int level, merged_file_t * files, int *maxNum)
{
    char path[1024];
    int cnt = 0;
    sstable_getpath(path, level);
    DIR * dir = opendir(path);
    if (dir == NULL)
    {
        perror(path);
        return -1;
    }
    struct dirent * ent;
    while ( (ent = readdir(dir)) != NULL && cnt < *maxNum)
    {
        int d_len = strlen(ent->d_name);
        if ( strncmp(ent->d_name, "sstable_", 8) == 0 && d_len > 15 &&
        strcmp(ent->d_name+(d_len-4), ".idx") == 0)
        {
            strncpy(files[cnt++].filename, ent->d_name, d_len-4); // subfix is ignore
        }
    }
    *maxNum = cnt;
    sstable_getScoreRange(level, files, cnt);
    qsort(files, cnt, sizeof(files[0]), compare_two_merged_file);

    return 0;
}
void sstable_getfilename(char * filename, int level, int subscript, int isIndex, int isTmp)
{
    if (isIndex)
    {
        if (isTmp)
        {
            sprintf(filename, "/data/SSTable%d/sstable_%09d.idx.tmp", level, subscript);
        }
        else
        {
            sprintf(filename, "/data/SSTable%d/sstable_%09d.idx", level, subscript);
        }
        
    }
    else
    {
        if (isTmp)
        {
            sprintf(filename, "/data/SSTable%d/sstable_%09d.dat.tmp", level, subscript);
        }
        else
        {
            sprintf(filename, "/data/SSTable%d/sstable_%09d.dat", level, subscript);
        }
    }
    
    
}
void sstable_getpath(char * path, int level)
{
    sprintf(path, "/data/SSTable%d", level);
}
int sstable_getOneRecord(FILE * fp, skiplist_buffer_t * score, skiplist_buffer_t * data)
{
    uint32_t fiedLen ;
    //score
    if (fread(&fiedLen, 1, sizeof(uint32_t), fp) != sizeof(uint32_t))
    {
        if (feof(fp)) {return 0;}
        perror("sstable_getOneRecord:read:");
        return -1;
    }
    fiedLen = ntohl(fiedLen);
    score->len = fiedLen;
    score->ptr = malloc(score->len);
    if (score->ptr == NULL)
    {
        fprintf(stderr, "failed to malloc\n");
        return -1;
    }
    if (fread(score->ptr, 1, score->len, fp) != score->len)
    {
        perror("sstable_getOneRecord:read:");
        skiplist_free_buffer(score);
        return -1;
    }
//data
    if (fread(&fiedLen, 1, sizeof(uint32_t), fp) != sizeof(uint32_t))
    {
        perror("sstable_getOneRecord:read:");
        return -1;
    }
    fiedLen = ntohl(fiedLen);
    data->len = fiedLen;
    data->ptr = malloc(data->len);
    if (data->ptr == NULL)
    {
        fprintf(stderr, "failed to malloc\n");
        skiplist_free_buffer(score);
        return -1;
    }
    if (fread(data->ptr, 1, data->len, fp) != data->len)
    {
        perror("sstable_getOneRecord:read:");
        skiplist_free_buffer(score);
        skiplist_free_buffer(data);
        return -1;
    }
    return 1;
}
static int find_position(const skiplist_buffer_t * score, const merged_file_t * files, int cnt, int * hole)
{
    int i;
    for (i = 0; i < cnt; ++i)
    {
        int iret1 = sstable_score_cmp(score, &files[i].beginScore);
        int iret2 = sstable_score_cmp(score, &files[i].endScore);
        if (iret1 < 0 && iret2 < 0)
        {
            *hole = i;
            return -1;
        }
        if (iret1 >=0 && iret2 <= 0)
        {
            return i;
        }
        if (iret1 > 0 && iret2 > 0)
        {
            continue;
        }
    }
    *hole = i;
    return -1;
}
static int fileMerge(const char * filename, int destLevel)
{
    char file1[1024];
    char file2[1024];
    char file3[1024];
    FILE * fp1 = NULL, *fp2 = NULL, *fp3 = NULL;

    static merged_file_t mergedFile[MAX_FILENUM_PER_LEVEL];
    memset(mergedFile, 0, sizeof(mergedFile));
    int maxNum = MAX_FILENUM_PER_LEVEL;
    sstable_getAllMergedFiles(destLevel, mergedFile, &maxNum);

    char path[1024];
    sstable_getpath(path, destLevel-1);
    sprintf(file1, "%s/%s.dat", path, filename);
    fp1 = fopen(file1, "rb");
    if (fp1 == NULL)
    {
        perror(file1);
        return -1;
    }
    fseek(fp1, 7, SEEK_CUR);

    int prev_hole = -1;
    int prev_position = -1;

    while (1)
    {
        skiplist_buffer_t score, data;
        memset(&score, 0, sizeof(score));
        memset(&data, 0, sizeof(data));
        int iret = sstable_getOneRecord(fp1, &score, &data);
        if (iret == 0)
        {
            break;
        }
        if (iret < 0)
        {
            fclose(fp1);
            skiplist_free_buffer(&score);
            skiplist_free_buffer(&data);
            return -1;
        }
        int hole = 0;
        int position = find_position(&score, mergedFile, maxNum, &hole);
        if (position >= 0) // valid position, merge two file
        {
            if (position != prev_position)
            {

            }
            else
            {
                
            }
        }
        else   // new file 
        {

        }

    }



    int i;
    for (i = 0; i < MAX_FILENUM_PER_LEVEL; ++i)
    {
        skiplist_free_buffer(&mergedFile[i].beginScore);
        skiplist_free_buffer(&mergedFile[i].endScore);
    }

    





}

int sstable_sink()
{
    int level = 0;
    for (level = 0; ; level++)
    {
        char path[1024];
        sstable_getpath(path, level);
        static char filenames[MAX_FILENUM_PER_LEVEL][1024];
        int maxNum = MAX_FILENUM_PER_LEVEL;
        //sstable_getAllFiles(level, filenames, &maxNum);
        if (maxNum > MAX_FILENUM_PER_LEVEL-100)
        {
            int i;
            for (i = 0; i < (maxNum - (MAX_FILENUM_PER_LEVEL-100) ); ++i )
            {
                char filename[1024];
                sprintf(filename, "%s/%s", path, filenames[i]);
                fileMerge(filename, level+1);
            }
            

        }
        break;


    }
    return 0;
}