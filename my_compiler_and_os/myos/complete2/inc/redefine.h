#ifndef REDEFINE_H_INCLUDED
#define REDEFINE_H_INCLUDED


#include <stdarg.h>

void memset(void * p, uint8_t v, uint32_t sz); /*用户态和内核态适用*/
void memcpy(void * dst, const void * src, uint32_t sz); /*用户态和内核态适用*/
void strcpy(char * dst, const char * src); /*用户态和内核态适用*/
int memcmp(const void * a, const void * b, size_t sz); /*用户态和内核态适用*/
int printf(const char *fmt, ...);  /*用户态适用*/
double strtod(const char *nptr, char **endptr); /*用户态和内核态适用*/
long int strtol(const char *nptr, char **endptr, int base); /*用户态和内核态适用*/
size_t strlen(const char *s);
int strncmp(const char * a, const char *b, size_t len);
int vsprintf(char *buf, size_t bufsz, const char *fmt, va_list args);
int snprintf(char * buf, size_t bufsz, const char *fmt, ...);

#endif
