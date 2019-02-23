#ifndef __COMMON_H__
#define __COMMON_H__

#define ID_MAX 25
#define QSTRING_BUF_MAX 1000
#define TRUE 	1
#define FALSE	0

#include "expr_type.h"

/*���ű����Ԫ��*/
typedef struct
{
	char tk_name[ID_MAX];
	int tk_type;
} Token;
/*α���ս������������*/
typedef struct 
{
	unsigned int label_false;	/*ѭ�����������������ʱ����תĿ��*/
	unsigned int label_true;	/*for ����������÷�*/
	unsigned int label_goto;	/*���һ��ѭ�������תĿ��*/
} Labels;
/*���̿������*/
typedef struct
{
	char fn_name[ID_MAX];
} FlowName;



#endif
