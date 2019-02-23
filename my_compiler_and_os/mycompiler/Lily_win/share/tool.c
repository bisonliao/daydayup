#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "tool.h"
#include "assert.h"
#include <ctype.h>
/*ȥ���ַ������Ŀո��tab��*/
void ltrim(char * p)
{
       int len;
       int start;
       int i;

       if (NULL == p)
       {
               return;
       }
       len = strlen(p);

       /*�ҵ���һ�����ǿո���ַ���λ��*/
       start = 0;
       while (	(p[start] == ' ' || p[start] == '\t') && start < len)
       {
               start++;
		}
       /*�������*/
       if (start > 0)
       {
               i = 0;
               while (start < len)
               {
                       p[i] = p[start];
                       i++;
                       start++;
               }
               p[i] = 0;
       }
}
/*ȥ���ַ����Ҳ�Ŀո��tab��*/
void rtrim(char *p)
{
       int len;

       if (NULL == p)
       {
               return;
       }

       len = strlen(p);
       len--;
       while ((p[len] == ' ' || p[len] == '\t')&&
               len >= 0)
       {
               len--;
       }
       p[len + 1] = 0;
}

void trim(char *p)
{
       if (NULL == p)
       {
               return;
       }
       ltrim(p);
       rtrim(p);
}
/*
*����Ƿ���ȫ�ֱ�����
*����ǣ�����1������ͨ��index���ظ�ȫ�ֱ����ı��;���򷵻�0
*/
int isGlobalID(const char* idname, int * index)
{
	int i;
	int len = strlen(idname);

	assert(NULL != index);
	if (idname[0] != 'g' || idname[1] != '_') /*g_...*/
	{
		return 0;
	}
	for (i = 2; i < len; i++)
	{
		if (!isdigit(idname[i]))
		{
			return 0;		
		}
	}
	i = atoi(idname + 2);
	if (i > 200)
	{
		return 0;
	}
	*index = i;
	return 1;
}

