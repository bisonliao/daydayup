#include "SSTable.h"
#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>

skiplist_cmpfunc_t sstable_score_cmp = skiplist_compare_score_default; 

int ssfile_init(ssfile_ctx * ctx, int level, int subscript, mode_t mode)
{
    int suc = 0;
    const int BUF_SIZE = (1024*1024*8);

    memset(ctx, 0, sizeof(ssfile_ctx));

    ctx->level = level;
    ctx->subscript = subscript;
    ctx->mode = mode;
  
    ctx->dat_fp = NULL;
    ctx->idx_fp = NULL;
    ctx->indexCnt = 0;
    ctx->dat_buf = malloc(BUF_SIZE);
    if (ctx->dat_buf == NULL)
    {
        goto init_end;
    }
    ctx->idx_buf = malloc(BUF_SIZE);
    if (ctx->idx_buf == NULL)
    {
        goto init_end;
    }

    char filename[1024];
    if (mode == O_APPEND)
    {
        sstable_getfilename(filename, level, subscript, 1, 1);
        ctx->idx_fp = fopen(filename, "w+b");
        if (ctx->idx_fp == NULL)
        {
            perror(filename);
            goto init_end;
        }
        fseek(ctx->idx_fp, sizeof(uint32_t), SEEK_SET);

        sstable_getfilename(filename, level, subscript, 0, 1);
        ctx->dat_fp = fopen(filename, "w+b");
        if (ctx->dat_fp == NULL)
        {
            perror(filename);
            goto init_end;
        }
        fwrite("sstable", 1, 7, ctx->dat_fp);
    }
    else if (mode == O_RDONLY)
    {
        sstable_getfilename(filename, level, subscript, 1, 0);
        ctx->idx_fp = fopen(filename, "rb");
        if (ctx->idx_fp == NULL)
        {
            perror(filename);
            goto init_end;
        }
        fseek(ctx->idx_fp, sizeof(uint32_t), SEEK_SET);

        sstable_getfilename(filename, level, subscript, 0, 0);
        ctx->dat_fp = fopen(filename, "rb");
        if (ctx->dat_fp == NULL)
        {
            perror(filename);

            goto init_end;
        }
        fseek(ctx->dat_fp, 7, SEEK_SET);
    }
    else
    {
        goto init_end;
    }
    setbuffer(ctx->idx_fp, ctx->idx_buf, BUF_SIZE);
    setbuffer(ctx->dat_fp, ctx->dat_buf, BUF_SIZE);
    suc = 1;
init_end:
    if (suc)
    {
        return 0;
    }
    skiplist_free_buffer(&ctx->maxScore);
    skiplist_free_buffer(&ctx->minScore);
    if (ctx->dat_buf)
    {
        free(ctx->dat_buf);
    }
    if (ctx->idx_buf)
    {
        free(ctx->idx_buf);
    }
    if (ctx->idx_fp)
    {
        fclose(ctx->idx_fp);
    }
    if (ctx->dat_fp)
    {
        fclose(ctx->dat_fp);
    }
    return -1;


    
}
int ssfile_init2(ssfile_ctx* ctx, int level, const char * filename, mode_t mode)
{
    int subscript;
    if (sscanf(filename, "sstable_%09d", &subscript) != 1)
    {
        return -1;
    }
    return ssfile_init(ctx, level, subscript, mode);

}
int ssfile_append(ssfile_ctx * ctx, const skiplist_buffer_t * score, const skiplist_buffer_t * data)
{
    if (ctx->mode != O_APPEND)
    {
        return -1;
    }
     // dat file structure:
    //  "sstable" + [score_len(4B) + score_value(?) + data_len(4B) + data_value(?)]
    // idx file structure:
    //  indexCnt(4B) + [score_len(4B) + score_value(?)+ offsetInDatFile(4B)] + IndexRangeInfo
    //  IndexRangeInfo:
    //  minScoreLen(4B) + minScoreData(?) + maxScoreLen(4B) + maxScoreData(?) + rangeInfoLen(4B) 
    
    uint32_t len;

    // write index file
    len = score->len;
    len = htonl(len);
    fwrite(&len, 1, sizeof(uint32_t), ctx->idx_fp);
    fwrite(score->ptr, 1, score->len, ctx->idx_fp);

    uint32_t offset = ftell(ctx->dat_fp);
    offset = htonl(offset);
    fwrite(&offset, 1, sizeof(uint32_t), ctx->idx_fp);

    // write data file
    len = score->len;
    len = htonl(len);
    fwrite(&len, 1, sizeof(uint32_t), ctx->dat_fp);
    fwrite(score->ptr, 1, score->len, ctx->dat_fp);
    len = data->len;
    len = htonl(len);
    fwrite(&len, 1, sizeof(uint32_t), ctx->dat_fp);
    fwrite(data->ptr, 1, data->len, ctx->dat_fp);

    ctx->indexCnt++;
    // update min / max score 
    if (ctx->minScore.len != 0)
    {
        if (sstable_score_cmp(&ctx->minScore, score) > 0)
        {
            skiplist_deep_copy_buffer(score, &ctx->minScore);
            //printf("update min score:%d\n", *(int*)ctx->minScore.ptr);
        }
    }
    else
    {
        skiplist_deep_copy_buffer(score, &ctx->minScore);
    }
    if (ctx->maxScore.len != 0)
    {
        if (sstable_score_cmp(&ctx->maxScore, score) < 0)
        {
            skiplist_deep_copy_buffer(score, &ctx->maxScore);
            //printf("update max score:%d\n", *(int*)ctx->maxScore.ptr);
        }
    }
    else
    {
        skiplist_deep_copy_buffer(score, &ctx->maxScore);
    }


}
int ssfile_final(ssfile_ctx * ctx, int sucFlag)
{
    if (ctx->mode == O_APPEND)
    {
        
        //write index count at the header of index file
        fseek(ctx->idx_fp, 0, SEEK_SET);
        uint32_t cnt = ctx->indexCnt;
        cnt = htonl(cnt);
        fwrite(&cnt, 1, sizeof(uint32_t), ctx->idx_fp);
        fseek(ctx->idx_fp, 0, SEEK_END);
        //write min/max score at the tail of index file 
        uint32_t rangeLen = 0;
        uint32_t len;
        
        len = ctx->minScore.len;
        len = htonl(len);
        fwrite(&len, 1, sizeof(uint32_t), ctx->idx_fp);
        fwrite(ctx->minScore.ptr, 1, ctx->minScore.len, ctx->idx_fp);

        len = ctx->maxScore.len;
        len = htonl(len);
        fwrite(&len, 1, sizeof(uint32_t), ctx->idx_fp);
        fwrite(ctx->maxScore.ptr, 1, ctx->maxScore.len, ctx->idx_fp);

        printf("%s:%d, %d, %d\n", __FUNCTION__, __LINE__, *(int*)ctx->minScore.ptr, *(int*)ctx->maxScore.ptr);

        rangeLen = sizeof(uint32_t) + ctx->minScore.len + sizeof(uint32_t) + ctx->maxScore.len + sizeof(uint32_t);
        //printf("range len:%u\n", rangeLen);
        rangeLen = htonl(rangeLen);

        fwrite(&rangeLen, 1, sizeof(uint32_t), ctx->idx_fp); 

    }
    else if (ctx->mode == O_RDONLY)
    {
        
    }


    skiplist_free_buffer(&ctx->maxScore);
    skiplist_free_buffer(&ctx->minScore);
    // order is important, close at first, then free buffer;
    if (ctx->idx_fp)
    {
        fclose(ctx->idx_fp);
    }
    if (ctx->dat_fp)
    {
        fclose(ctx->dat_fp);
    }

    if (ctx->dat_buf)
    {
        free(ctx->dat_buf);
    }
    if (ctx->idx_buf)
    {
        free(ctx->idx_buf);
    }

    if (ctx->mode == O_APPEND && sucFlag)
    {
        char newfilename[1024];
        char filename[1024];
        sstable_getfilename(filename, ctx->level, ctx->subscript, 1, 1);
        sstable_getfilename(newfilename, ctx->level, ctx->subscript, 1, 0);
        rename(filename, newfilename);
        sstable_getfilename(filename, ctx->level, ctx->subscript, 0, 1);
        sstable_getfilename(newfilename, ctx->level, ctx->subscript, 0, 0);
        rename(filename, newfilename);
    }

}
int ssfile_read(ssfile_ctx * ctx, skiplist_buffer_t * score, skiplist_buffer_t * data)
{
    skiplist_free_buffer(score);
    skiplist_free_buffer(data);
    if (ctx->mode != O_RDONLY)
    {
        return -1;
    }
    uint32_t fiedLen ;
    //score
    if (fread(&fiedLen, 1, sizeof(uint32_t), ctx->dat_fp) != sizeof(uint32_t))
    {
        if (feof(ctx->dat_fp)) {return 0;} // file end
        perror("ssfile_read:read:");
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
    if (fread(score->ptr, 1, score->len, ctx->dat_fp) != score->len)
    {
        perror("ssfile_read:read:");
        skiplist_free_buffer(score);
        return -1;
    }
//data
    if (fread(&fiedLen, 1, sizeof(uint32_t), ctx->dat_fp) != sizeof(uint32_t))
    {
        perror("ssfile_read:read:");
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
    if (fread(data->ptr, 1, data->len, ctx->dat_fp) != data->len)
    {
        perror("ssfile_read:read:");
        skiplist_free_buffer(score);
        skiplist_free_buffer(data);
        return -1;
    }
    return 1;
    
}
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

static int find_position(const skiplist_buffer_t * score, const merged_file_t * files, int cnt, int * hole)
{
    int i;
    for (i = 0; i < cnt; ++i)
    {
        int iret1 = sstable_score_cmp(score, &files[i].beginScore);
        int iret2 = sstable_score_cmp(score, &files[i].endScore);
        //printf("cnt=%d, %d, %d,%d\n",cnt, *(int*)files[i].beginScore.ptr, *(int*)files[i].endScore.ptr, *(int*)score->ptr);
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
int sstable_getNextFileIndex(int level, int start)
{
 
    int i;
    for (i = start; i< 1000000; ++i)
    {
        char filename[1024];
        sstable_getfilename(filename, level, i, 1, 0);
      
        if (access(filename, F_OK) == 0 )
        {
            continue;
        }
        else
        {
            break;
        }
    }
    return i;
  
}
static int fileMerge1(
    ssfile_ctx * toMerge, 
    ssfile_ctx* matchedFile, 
    ssfile_ctx * resultFile, 
    skiplist_buffer_t * firstScore,
    skiplist_buffer_t * firstData)
{
    skiplist_buffer_t score2, data2;
    memset(&score2, 0, sizeof(score2));
    memset(&data2, 0, sizeof(data2));
    int flag = -1; // -1: failed  0: success  1: toMerge ended

    while (1)
    {
        
        int readResult = ssfile_read(matchedFile, &score2, &data2);
        if (readResult == 0)
        {
            flag = 0;
            break;
        }
        if (readResult < 0)
        {
            fprintf(stderr, "read record failed.\n");
            goto endFlag;
        }
        if (sstable_score_cmp(firstScore, &score2) <= 0)
        {
            ssfile_append(resultFile, firstScore, firstData);
        }
        else
        {
            ssfile_append(resultFile, &score2, &data2);
        }

        
        readResult = ssfile_read(toMerge, firstScore, firstData);
        if (readResult == 0)
        {
            flag = 1;
            break;
        }
        if (readResult < 0)
        {
            fprintf(stderr, "read record failed. \n");
            goto endFlag;
        }
    }
    if (flag == 1) // toMerge ended 
    {
        while (1)
        {
            int readResult = ssfile_read(matchedFile, &score2, &data2); //matchedFile ended too
            if (readResult == 0)
            {
                break;
            }
            if (readResult < 0)
            {
                fprintf(stderr, "read record failed. \n");
                goto endFlag;
            }
            ssfile_append(resultFile, &score2, &data2);
        }
        
    }
endFlag:
    

    skiplist_free_buffer(&score2);
    skiplist_free_buffer(&data2);

    return flag;
}
int fileMerge(const char * filename, int destLevel)
{
    char file[1024];
    char path[1024];
    int suc = 0;
    

    static merged_file_t mergedFile[MAX_FILENUM_PER_LEVEL];
    memset(mergedFile, 0, sizeof(mergedFile));
    int maxNum = MAX_FILENUM_PER_LEVEL;
    sstable_getAllMergedFiles(destLevel, mergedFile, &maxNum);

    
    ssfile_ctx toMerge;
    ssfile_init2(&toMerge, destLevel-1, filename, O_RDONLY);

    skiplist_buffer_t score, data;
    memset(&score, 0, sizeof(score));
    memset(&data, 0, sizeof(data));
    while (1)
    {
        int readResult = ssfile_read(&toMerge, &score, &data);
        if (readResult == 0)
        {
            suc = 1;
            break;
        }
        if (readResult < 0)
        {
            fprintf(stderr, "read record failed. %s\n", filename);
            break;
        }
        int hole = 0;
        int position = find_position(&score, mergedFile, maxNum, &hole);
        printf("find position:%d,%d, score:%d\n", position, hole, *(int*)score.ptr);
        //return 0;
        if (position >= 0) // valid position, merge two file
        {
            ssfile_ctx matchedFile, resultFile;
            ssfile_init2(&matchedFile, destLevel, mergedFile[position].filename, O_RDONLY);
            ssfile_init2(&resultFile, destLevel, mergedFile[position].filename, O_APPEND);
            printf("merge to  file:%d,%s.dat\n", destLevel, mergedFile[position].filename);
            int iret = fileMerge1(&toMerge, &matchedFile, &resultFile, &score, &data);
            ssfile_final(&matchedFile, 1);
            ssfile_final(&resultFile, iret >= 0);
            
            if (iret == 1) // toMerge ended
            {
                suc = 1;
                break;
            }
            if (iret < 0)
            {
                break;
            }
            
        }
        else // new file
        {

            ssfile_ctx  resultFile;
            int subscript = sstable_getNextFileIndex(destLevel, 0);
            ssfile_init(&resultFile, destLevel, subscript, O_APPEND);
            printf("new file:%d,%d\n", destLevel, subscript);
            int flag = 0;
            // write to new file
            while (sstable_score_cmp(&score, &mergedFile[hole].beginScore) <= 0)
            {
                ssfile_append(&resultFile, &score, &data);
                readResult = ssfile_read(&toMerge, &score, &data);
                if (readResult == 0)
                {
                    flag = 1;
                    break;
                }
                if (readResult < 0)
                {
                    flag = -1;
                    fprintf(stderr, "read record failed. %s\n", filename);
                    break;
                }
            }
            ssfile_final(&resultFile, flag >= 0);
            if (flag == 1)
            {
                suc = 1;
                break;
            }
            if (flag == -1)
            {
                break;
            }
            
        }
    }
    skiplist_free_buffer(&score);
    skiplist_free_buffer(&data);

    int i;
    for (i = 0; i < MAX_FILENUM_PER_LEVEL; ++i)
    {
        skiplist_free_buffer(&mergedFile[i].beginScore);
        skiplist_free_buffer(&mergedFile[i].endScore);
    }
    return 0;
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