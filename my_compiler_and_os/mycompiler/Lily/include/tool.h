#ifndef __TOOL_H__
#define __TOOL_H__

#ifdef __cplusplus
extern "C" {
#endif 

void rtrim(char *p);
void ltrim(char *p);
void trim(char *p);
int isGlobalID(const char* idname, int * index);

#ifdef __cplusplus
}
#endif 

#endif
