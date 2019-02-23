#ifndef REDEFINE_H_INCLUDED
#define REDEFINE_H_INCLUDED


#include <stdarg.h>

void memset(void * p, uint8_t v, uint32_t sz); /*�û�̬���ں�̬����*/
void memcpy(void * dst, const void * src, uint32_t sz); /*�û�̬���ں�̬����*/
void strcpy(char * dst, const char * src); /*�û�̬���ں�̬����*/
int memcmp(const void * a, const void * b, size_t sz); /*�û�̬���ں�̬����*/
int printf(const char *fmt, ...);  /*�û�̬����*/
double strtod(const char *nptr, char **endptr); /*�û�̬���ں�̬����*/
long int strtol(const char *nptr, char **endptr, int base); /*�û�̬���ں�̬����*/
size_t strlen(const char *s);
int strncmp(const char * a, const char *b, size_t len);
int vsprintf(char *buf, size_t bufsz, const char *fmt, va_list args);
int snprintf(char * buf, size_t bufsz, const char *fmt, ...);

#endif
