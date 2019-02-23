#ifndef __COMMON_H__
#define __COMMON_H__

#define ID_MAX 25
#define OPERATOR_MAX 15
#define TRUE 1
#define FALSE 0

/*
*一条指令的类型
*比如 VAR aaa INTEGER
*/
typedef struct 
{
	char inst_action[OPERATOR_MAX];
	char inst_operant1[ID_MAX];
	char inst_operant2[ID_MAX];
} INSTRUCT;


#endif
