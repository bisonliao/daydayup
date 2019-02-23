#ifndef API_H_INCLUDED
#define API_H_INCLUDED


#include "struct.h"
#include "const_def.h"
/*供应用程序使用的两个系统调用*/
uint32_t _set_ticks(uint32_t v);
uint32_t _get_ticks();
int32_t _hd(uint32_t abs_sector, void * buf, uint32_t cmd);
uint32_t _time();
void _sync();
int32_t  _open(const char*, uint32_t);
int32_t  _mkdir(const char*);
int32_t  _mkfile(const char*);
int32_t  _rmfile(const char*);
int32_t  _rmdir(const char*);
int32_t  _read(uint32_t, char*, uint32_t);
int32_t  _write(uint32_t, const char*, uint32_t);
int32_t  _cout(const char*, size_t);
int32_t  _fork();
int32_t  _exec(const char *, char * const argv[]);
int32_t  _exit();
int32_t  _cin(char * , uint32_t);
int32_t  _access(const char * , uint32_t);
int32_t  _wait(int pid, int * status);

void proc_A();
void proc_B();
#endif
