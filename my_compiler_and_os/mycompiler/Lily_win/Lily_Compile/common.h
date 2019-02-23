#ifndef __COMMON_H__
#define __COMMON_H__

#define ID_MAX 25
#define QSTRING_BUF_MAX 1000
#define TRUE 	1
#define FALSE	0

#include "expr_type.h"

/*符号表里的元素*/
typedef struct
{
	char tk_name[ID_MAX];
	int tk_type;
} Token;
/*伪非终结符的属性类型*/
typedef struct 
{
	unsigned int label_false;	/*循环体进入条件不满足时的跳转目标*/
	unsigned int label_true;	/*for 语句中特殊用法*/
	unsigned int label_goto;	/*完成一轮循环后的跳转目标*/
} Labels;
/*流程块的名字*/
typedef struct
{
	char fn_name[ID_MAX];
} FlowName;



#endif
