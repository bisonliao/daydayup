#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Variable.h"
#include "Variable_Stack.h"
#include "instruct_list.h"
#include "int_Stack.h"
#include "tool.h"
#include "Run_Fun.h"
#ifdef _MAKE_DLL
#include "lily_run.h"
#endif

static Variable_Stack	operation_stack;	/*����ջ*/
static Variable_Stack	data_stack;		    /*����ջ*/	
static int_Stack		varnum_stack;		/*�ֲ���������ջ*/
static int_Stack		recall_stack;		/*���û���ջ*/
static Instruct_List	instruct_list;		/*ָ�����*/
static Variable			global_var[201];	/*ȫ�ֱ���g_0 ... g_200*/

int ExecuteInstruct(const INSTRUCT & inst, char *errmsg);
bool FindVarByName(const AnsiString& name,Variable& ele);
bool ModifyVarByName(const AnsiString &name, const Variable& ele);

#undef SHOW_STEP
#if defined(_DEBUG)
	#undef SHOW_STEP
	#define SHOW_STEP printf("ִ��%s�ĵ�%d��...\n", __FILE__, __LINE__);
	//#define SHOW_STEP ;
#else
	#define SHOW_STEP ;
#endif 

#if !defined(_MAKE_DLL)

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		printf("usage:%s <file to execute>\n", argv[0]);
		return 0;	
	}
	/*��ʼ��global_var����*/
	for (int i = 0; i < (sizeof(global_var)/sizeof(Variable)); ++i)
	{
		global_var[i].setType(TYPE_MEMBLOCK);	
	}
	
	/*װ��ָ��ͳ����ַ���*/
	char errmsg[100];
	if (instruct_list.Initialize(argv[1], errmsg) != 0)
	{
		fprintf(stderr, "װ��ָ��ͳ����ַ���ʧ�ܣ�����ԭ��%s\n", errmsg);
		return -1;
	}
	/*ִ��ָ��*/
	instruct_list.BeginExecute();	
	while (1)
	{
		INSTRUCT inst;
		if ( instruct_list.NextInstruct(inst) != 0)
		{
			fprintf(stderr, "ȡ��һ��ָ���!\n");
			return -1;
		}
		if (strcmp(inst.inst_action, "HALT") == 0)
		{
			break;
		}
		if (ExecuteInstruct(inst, errmsg) != 0)
		{
			fprintf(stderr, "ִ��ָ�����,ԭ��:%s\n", errmsg);
			return -1;
		}
	}
	return 0;
}
#else
int ExecuteFile(const char * filename, 
				const char* InputBuffer, unsigned int InputBufferSize,
				char* OutputBuffer, unsigned int *OutputBufferSize,
				char * errmsg)
{
	if (NULL == filename || errmsg == NULL || NULL == InputBuffer || OutputBuffer == NULL
		|| NULL == OutputBufferSize)
	{
		return -1;	
	}
	/*װ��ָ��ͳ����ַ���*/
	char errmsg1[100];
	if (instruct_list.Initialize(filename, errmsg1) != 0)
	{
		sprintf(errmsg, "װ��ָ��ͳ����ַ���ʧ�ܣ�����ԭ��%s\n", errmsg1);
		return -1;
	}
	/*��ʼ��global_var����*/
	for (int i = 0; i < (sizeof(global_var)/sizeof(Variable)); ++i)
	{
		global_var[i].setType(TYPE_MEMBLOCK);	
	}
	/*��ʼ��ȫ�ֱ���g_0��Ϊ�����ڴ���*/
	MemBlock memblock;
	if (memblock.SetValue(InputBuffer, InputBufferSize) != 0)
	{
		sprintf(errmsg, "�����ڴ�������ʧ��!\n");
		return -1;	
	}
	global_var[0].setMemBlock(memblock);
	
	/*ִ��ָ��*/
	instruct_list.BeginExecute();	
	while (1)
	{
		INSTRUCT inst;
		if ( instruct_list.NextInstruct(inst) != 0)
		{
			sprintf(errmsg, "ȡ��һ��ָ���!\n");
			return -1;
		}
		if (strcmp(inst.inst_action, "HALT") == 0)
		{
			break;
		}
		if (ExecuteInstruct(inst, errmsg1) != 0)
		{
			sprintf(errmsg, "ִ��ָ�����,ԭ��:%s\n", errmsg1);
			return -1;
		}
	}
	/*��g_1���������Ϊ�����ڴ�*/
	memblock = global_var[0].getMemBlock();
	if (memblock.GetSize() < *OutputBufferSize)
	{
		*OutputBufferSize = memblock.GetSize();
	}
	memcpy(OutputBuffer, memblock.GetBufferPtr(), *OutputBufferSize);
	return 0;
}
#endif
int ExecuteInstruct(const INSTRUCT & inst, char *errmsg)
{
	#ifdef _DEBUG
	printf("...Instruct:[%s][%s][%s]\n", inst.inst_action,
				inst.inst_operant1,
				inst.inst_operant2);
				
	#endif
	if (strcmp(inst.inst_action, "LABEL") == 0)
	{
		return 0;
	}
	if (strcmp(inst.inst_action, "DEPTH") == 0)
	{
		varnum_stack.push(0);
		return 0;
	}
	if (strcmp(inst.inst_action, "_DEPTH") == 0)
	{
		int varnum;
		Variable var;
		varnum_stack.pop(varnum);
		for (int i = 0; i < varnum; i++)
		{
			data_stack.pop(var);
		}
		return 0;
	}
	if (strcmp(inst.inst_action, "VAR") == 0)
	{
		/* VAR n_count INTEGER */
		Variable var;
		var.setName(inst.inst_operant1);
		if (strcmp(inst.inst_operant2, "INTEGER") == 0)
		{
			var.setType(TYPE_INTEGER);
		}
		else if (strcmp(inst.inst_operant2, "FLOAT") == 0)
		{
			var.setType(TYPE_FLOAT);
		}
		else if (strcmp(inst.inst_operant2, "MEMBLOCK") == 0)
		{
			var.setType(TYPE_MEMBLOCK);
		}
		else 
		{
			var.setType(TYPE_STRING);
		}
		data_stack.push(var);

		/*���ӱ�������*/
		int varnum;
		varnum_stack.pop(varnum);
		varnum++;
		varnum_stack.push(varnum);

		return 0;
	}
	if (strcmp(inst.inst_action, "PUSH") == 0)
	{
		/* 	
		*	PUSH #1
		*	PUSH #1.2
		*	PUSH %1
		*	PUSH n_count 
		*/	
		if (inst.inst_operant1[0] == '#')	/*������������������*/
		{
			Variable var;
			if (strchr(inst.inst_operant1, '.') == NULL)	/*����*/
			{
				var.setType(TYPE_INTEGER);
				var.setInteger(atoi(inst.inst_operant1 + 1));
			}
			else	/*������*/
			{
				var.setType(TYPE_FLOAT);
				var.setFloat(atof(inst.inst_operant1 + 1));
			}
			operation_stack.push(var);
			return 0;
		}
		if (inst.inst_operant1[0] == '%')	/*�����ַ���*/
		{
			int index = atoi(inst.inst_operant1 + 1);
			Variable var;
			AnsiString s;
			if (instruct_list.GetConstString(index, s) != 0)
			{
				sprintf(errmsg, "������Ϊ[%d]�ĳ����ַ���������!", index);
				return -1;
			}
			var.setType(TYPE_STRING);
			var.setString(s);
			
			operation_stack.push(var);
			return 0;
		}
		/*����*/
		int varnum;
		AnsiString name;
		Variable v;

		name = inst.inst_operant1;
		if (FindVarByName(name, v) != TRUE)
		{
			sprintf(errmsg, "���ұ���[%s]ʧ��!", name.c_str());
			return -1;
		}
	#ifdef _DEBUG
		/*
		if (v.getType() == TYPE_STRING)
		{
			printf("[%s][%d][%s]=[%s]\n", 
					__FILE__,
					__LINE__,
					v.getName().c_str(),
					v.getString().c_str());
		}
		if (v.getType() == TYPE_INTEGER)
		{
			printf("[%s][%d][%s]=[%d]\n", 
					__FILE__,
					__LINE__,
					v.getName().c_str(),
					v.getInteger());
		}
		if (v.getType() == TYPE_FLOAT)
		{
			printf("[%s][%d][%s]=[%d]\n", 
					__FILE__,
					__LINE__,
					v.getName().c_str(),
					v.getFloat());
		}
		*/
	#endif
		operation_stack.push(v);
		return 0;
	}
	if (strcmp(inst.inst_action, "MUL") == 0)
	{
		Variable left, right,result;
		operation_stack.pop(right);
		operation_stack.pop(left);
		if (left.getType() == TYPE_INTEGER)
		{
			if (right.getType() == TYPE_INTEGER)
			{
				result.setType(TYPE_INTEGER);
				result.setInteger(left.getInteger() * right.getInteger());
			}
			else if (right.getType() == TYPE_FLOAT)
			{
				result.setType(TYPE_FLOAT);
				result.setFloat(left.getInteger() * right.getFloat());
			}
			else
			{
				sprintf(errmsg, "MULָ����Ҳ��������Ͳ���ȷ!");
				return -1;
			}
		}
		else if (left.getType() == TYPE_FLOAT)
		{
			if (right.getType() == TYPE_INTEGER)
			{
				result.setType(TYPE_FLOAT);
				result.setFloat(left.getFloat() * right.getInteger());
			}
			else if (right.getType() == TYPE_FLOAT)
			{
				result.setType(TYPE_FLOAT);
				result.setFloat(left.getFloat() * right.getFloat());
			}
			else
			{
				sprintf(errmsg, "MULָ����Ҳ��������Ͳ���ȷ!");
				return -1;
			}

		}
		else
		{
			sprintf(errmsg, "MULָ�������������Ͳ���ȷ!");
			return -1;
		}
		operation_stack.push(result);
		return 0;
	}
	if (strcmp(inst.inst_action, "ADD") == 0)
	{
		Variable left, right,result;
		operation_stack.pop(right);
		operation_stack.pop(left);
		if (left.getType() == TYPE_STRING && right.getType() == TYPE_STRING)
		{
			result.setType(TYPE_STRING);
			AnsiString as("");
			as.concat(left.getString());
			as.concat(right.getString());
			result.setString(as);
		}
		else  if (left.getType() != TYPE_STRING && right.getType() != TYPE_STRING)
		{
			if (left.getType() == TYPE_INTEGER)
			{
				if (right.getType() == TYPE_INTEGER)
				{
					result.setType(TYPE_INTEGER);
					result.setInteger(left.getInteger() + right.getInteger());
				}
				else
				{
					result.setType(TYPE_FLOAT);
					result.setFloat(left.getInteger() + right.getFloat());
				}
			}
			else
			{
				if (right.getType() == TYPE_INTEGER)
				{
					result.setType(TYPE_FLOAT);
					result.setFloat(left.getFloat() + right.getInteger());
				}
				else
				{
					result.setType(TYPE_FLOAT);
					result.setFloat(left.getFloat() + right.getFloat());
				}
			}
		}
		else
		{
			sprintf(errmsg, "ADDָ��Ĳ��������Ͳ���ȷ!");
			return -1;
		}
		operation_stack.push(result);
		return 0;
	}
	if (strcmp(inst.inst_action, "MOD") == 0)
	{
		Variable left, right,result;
		operation_stack.pop(right);
		operation_stack.pop(left);
		if (left.getType() != TYPE_INTEGER || right.getType() != TYPE_INTEGER)
		{
			sprintf(errmsg, "MODָ��Ĳ��������Ͳ���ȷ!");
			return -1;
		}
		result.setType(TYPE_INTEGER);
		result.setInteger(left.getInteger() % right.getInteger());
		operation_stack.push(result);
		return 0;
	}
	if (strcmp(inst.inst_action, "SUB") == 0)
	{
		Variable left, right,result;
		operation_stack.pop(right);
		operation_stack.pop(left);
		if (left.getType() == TYPE_INTEGER)
		{
			if (right.getType() == TYPE_INTEGER)
			{
				result.setType(TYPE_INTEGER);
				result.setInteger(left.getInteger() - right.getInteger());
			}
			else if (right.getType() == TYPE_FLOAT)
			{
				result.setType(TYPE_FLOAT);
				result.setFloat(left.getInteger() - right.getFloat());
			}
			else
			{
				sprintf(errmsg, "SUBָ����Ҳ��������Ͳ���ȷ!");
				return -1;
			}
		}
		else if (left.getType() == TYPE_FLOAT)
		{
			if (right.getType() == TYPE_INTEGER)
			{
				result.setType(TYPE_FLOAT);
				result.setFloat(left.getFloat() - right.getInteger());
			}
			else if (right.getType() == TYPE_FLOAT)
			{
				result.setType(TYPE_FLOAT);
				result.setFloat(left.getFloat() - right.getFloat());
			}
			else
			{
				sprintf(errmsg, "SUBָ����Ҳ��������Ͳ���ȷ!");
				return -1;
			}

		}
		else
		{
			sprintf(errmsg, "SUBָ�������������Ͳ���ȷ!");
			return -1;
		}
		operation_stack.push(result);
		return 0;
	}
	if (strcmp(inst.inst_action, "DIV") == 0)
	{
		Variable left, right,result;
		operation_stack.pop(right);
		operation_stack.pop(left);
		if ( (right.getType() == TYPE_INTEGER && right.getInteger() == 0)
			|| (right.getType() == TYPE_FLOAT && right.getFloat() == 0.0))
		{
			sprintf(errmsg, "����Ϊ0!");
			return -1;
		}
		if (left.getType() == TYPE_INTEGER)
		{
			if (right.getType() == TYPE_INTEGER)
			{
				result.setType(TYPE_INTEGER);
				result.setInteger(left.getInteger() / right.getInteger());
			}
			else if (right.getType() == TYPE_FLOAT)
			{
				result.setType(TYPE_FLOAT);
				result.setFloat(left.getInteger() / right.getFloat());
			}
			else
			{
				sprintf(errmsg, "DIVָ����Ҳ��������Ͳ���ȷ!");
				return -1;
			}
		}
		else if (left.getType() == TYPE_FLOAT)
		{
			if (right.getType() == TYPE_INTEGER)
			{
				result.setType(TYPE_FLOAT);
				result.setFloat(left.getFloat() / right.getInteger());
			}
			else if (right.getType() == TYPE_FLOAT)
			{
				result.setType(TYPE_FLOAT);
				result.setFloat(left.getFloat() / right.getFloat());
			}
			else
			{
				sprintf(errmsg, "DIVָ����Ҳ��������Ͳ���ȷ!");
				return -1;
			}

		}
		else
		{
			sprintf(errmsg, "DIVָ�������������Ͳ���ȷ!");
			return -1;
		}
		operation_stack.push(result);
		return 0;
	}
	if (strcmp(inst.inst_action, "SAV") == 0)
	{
		AnsiString name;
		Variable v;
		
		operation_stack.pop(v);
	
		/*���ݱ������ֺ͵�ǰģ��ı�������������ջ�в��Ҳ��޸ı���*/
		name = inst.inst_operant1;
		v.setName(name);
		if (ModifyVarByName(name,  v) != TRUE)
		{
			sprintf(errmsg, "���޸ı���[%s]ʧ��!", name.c_str());
			return -1;
		}
	
		return 0;
	}
	if (strcmp(inst.inst_action, "GOTO") == 0)
	{
		int index = atoi(inst.inst_operant2 + 1);
		instruct_list.JmpToInstruct(index);
		return 0;
	}
	if (strcmp(inst.inst_action, "GOTOTRUE") == 0)
	{
		/* GOTOTRUE L_1 @16 */
		Variable v;
		operation_stack.pop(v);
		if (v.getType() == TYPE_INTEGER)
		{
			if (v.getInteger() != 0)
			{
				int index = atoi(inst.inst_operant2 + 1);
				instruct_list.JmpToInstruct(index);
			}
		}
		else
		{
			sprintf(errmsg, "GOTOTRUEָ���ջ�����������ʹ���!");
			return -1;
		}
		return 0;
	}
	if (strcmp(inst.inst_action, "GOTOFALSE") == 0)
	{
		/* GOTOFALSE L_1 @16 */
		Variable v;
		operation_stack.pop(v);
		if (v.getType() == TYPE_INTEGER)
		{
			if (v.getInteger() == 0)
			{
				int index = atoi(inst.inst_operant2 + 1);
				instruct_list.JmpToInstruct(index);
			}
		}
		else
		{
			sprintf(errmsg, "GOTOTRUEָ���ջ�����������ʹ���!");
			return -1;
		}
		return 0;
	}
	if (strcmp(inst.inst_action, "DUP") == 0)
	{
		Variable v;
		operation_stack.peek(v);
		operation_stack.push(v);
		return 0;
	}
	if (strcmp(inst.inst_action, "RECALL") == 0)
	{
		int index;
		recall_stack.pop(index);
		instruct_list.JmpToInstruct(index);
		return 0;
	}
	if (strcmp(inst.inst_action, "AND") == 0)
	{
		Variable left, right, result;
		operation_stack.pop(right);
		operation_stack.pop(left);
		result.setType(TYPE_INTEGER);
		if (left.getInteger() != 0 && right.getInteger() != 0)
		{
			result.setInteger(1);
		}
		else
		{
			result.setInteger(0);
		}
		operation_stack.push(result);
		return 0;
	}
	if (strcmp(inst.inst_action, "CALL") == 0)
	{
		int funnum = atoi(inst.inst_operant1);

		Variable argc;
		operation_stack.pop(argc);
		

		Variable* argv = new Variable[argc.getInteger() + 1];
		if (argv == NULL)
		{
			sprintf(errmsg, "���к���ǰ��̬����ռ�ʧ��!");
			return -1;
		}
		
		/*�����Ĳ���������argv[1]...argv[argc]*/
		for (int i = argc.getInteger(); i >= 1; i--)
		{
			operation_stack.pop(argv[i]);
		
		}
		
		if (Run_Fun(funnum, argc.getInteger(), argv) != 0)
		{
			fprintf(stderr, "���Ϊ%d�ĺ���ִ�г����������ͻ��������ȷ!\n",
				funnum);
			delete[] argv;
			return -1;
		}
		
		
		/*����ֵ������argv[0]*/	
		if (operation_stack.push(argv[0]) != TRUE)
		{
			fprintf(stderr, "����ִ�н����ջʧ��!\n",
				funnum);
			delete[] argv;
			return -1;
		}
			
		delete[] argv;
		return 0;
	}
	if (strcmp(inst.inst_action, "EQ") == 0)
	{
		Variable left, right, result;

		operation_stack.pop(right);
		operation_stack.pop(left);

		result.setType(TYPE_INTEGER);
		result.setInteger(0);

		if (left.getType() == TYPE_STRING && right.getType() == TYPE_STRING)
		{
			if (left.getString() == right.getString())
			{
				result.setInteger(1);
			}
		}
		else if (left.getType() == TYPE_MEMBLOCK && right.getType() == TYPE_MEMBLOCK)
		{
			if (left.getMemBlock() == right.getMemBlock())
			{
				result.setInteger(1);
			}
		}
		else if ( (left.getType() == TYPE_INTEGER || left.getType() == TYPE_FLOAT) &&
			(right.getType() == TYPE_INTEGER || right.getType() == TYPE_FLOAT))
		{
			if (left.getType() == TYPE_INTEGER)
			{
				if (right.getType() == TYPE_INTEGER)
				{
					if (left.getInteger() == right.getInteger())
					{
						result.setInteger(1);
					}
				}
				else
				{
					if (left.getInteger() == right.getFloat())
					{
						result.setInteger(1);
					}
				}
			}
			else
			{
				if (right.getType() == TYPE_INTEGER)
				{
					if (left.getFloat() == right.getInteger())
					{
						result.setInteger(1);
					}
				}
				else
				{
					if (left.getFloat() == right.getFloat())
					{
						result.setInteger(1);
					}
				}
			}
		}
		else
		{
			sprintf(errmsg, "EQָ����������������Ͳ���ȷ!");
			return -1;
		}
		operation_stack.push(result);
		return 0;
	}
	if (strcmp(inst.inst_action, "GE") == 0)
	{
		Variable left, right, result;

		operation_stack.pop(right);
		operation_stack.pop(left);

		result.setType(TYPE_INTEGER);
		result.setInteger(0);

		if (left.getType() == TYPE_INTEGER)
		{
			if (right.getType() == TYPE_INTEGER)
			{
				if (left.getInteger() >= right.getInteger())
				{
					result.setInteger(1);
				}
			}
			else
			{
				if (left.getInteger() >= right.getFloat())
				{
					result.setInteger(1);
				}
			}
		}
		else
		{
			if (right.getType() == TYPE_INTEGER)
			{
				if (left.getFloat() >= right.getInteger())
				{
					result.setInteger(1);
				}
			}
			else
			{
				if (left.getFloat() >= right.getFloat())
				{
					result.setInteger(1);
				}
			}
		}
		operation_stack.push(result);	
		return 0;
	}
	if (strcmp(inst.inst_action, "GT") == 0)
	{
		Variable left, right, result;

		operation_stack.pop(right);
		operation_stack.pop(left);

		result.setType(TYPE_INTEGER);
		result.setInteger(0);

		if (left.getType() == TYPE_INTEGER)
		{
			if (right.getType() == TYPE_INTEGER)
			{
				if (left.getInteger() > right.getInteger())
				{
					result.setInteger(1);
				}
			}
			else
			{
				if (left.getInteger() > right.getFloat())
				{
					result.setInteger(1);
				}
			}
		}
		else
		{
			if (right.getType() == TYPE_INTEGER)
			{
				if (left.getFloat() > right.getInteger())
				{
					result.setInteger(1);
				}
			}
			else
			{
				if (left.getFloat() > right.getFloat())
				{
					result.setInteger(1);
				}
			}
		}
		operation_stack.push(result);	
		return 0;
	}
	if (strcmp(inst.inst_action, "LE") == 0)
	{
		Variable left, right, result;

		operation_stack.pop(right);
		operation_stack.pop(left);

		result.setType(TYPE_INTEGER);
		result.setInteger(0);

		if (left.getType() == TYPE_INTEGER)
		{
			if (right.getType() == TYPE_INTEGER)
			{
				if (left.getInteger() <= right.getInteger())
				{
					result.setInteger(1);
				}
			}
			else
			{
				if (left.getInteger() <= right.getFloat())
				{
					result.setInteger(1);
				}
			}
		}
		else
		{
			if (right.getType() == TYPE_INTEGER)
			{
				if (left.getFloat() <= right.getInteger())
				{
					result.setInteger(1);
				}
			}
			else
			{
				if (left.getFloat() <= right.getFloat())
				{
					result.setInteger(1);
				}
			}
		}
		operation_stack.push(result);	
		return 0;
	}
	if (strcmp(inst.inst_action, "LT") == 0)
	{
		Variable left, right, result;

		operation_stack.pop(right);
		operation_stack.pop(left);

		result.setType(TYPE_INTEGER);
		result.setInteger(0);

		if (left.getType() == TYPE_INTEGER)
		{
			if (right.getType() == TYPE_INTEGER)
			{
				if (left.getInteger() < right.getInteger())
				{
					result.setInteger(1);
				}
			}
			else
			{
				if (left.getInteger() < right.getFloat())
				{
					result.setInteger(1);
				}
			}
		}
		else
		{
			if (right.getType() == TYPE_INTEGER)
			{
				if (left.getFloat() < right.getInteger())
				{
					result.setInteger(1);
				}
			}
			else
			{
				if (left.getFloat() < right.getFloat())
				{
					result.setInteger(1);
				}
			}
		}
		operation_stack.push(result);	
		return 0;
	}
	if (strcmp(inst.inst_action, "NE") == 0)
	{
		Variable left, right, result;

		operation_stack.pop(right);
		operation_stack.pop(left);

		result.setType(TYPE_INTEGER);
		result.setInteger(1);

		if (left.getType() == TYPE_STRING && right.getType() == TYPE_STRING)
		{
			if (left.getString() == right.getString())
			{
				result.setInteger(0);
			}
		}
		else if (left.getType() == TYPE_MEMBLOCK && right.getType() == TYPE_MEMBLOCK)
		{
			if (left.getMemBlock() == right.getMemBlock())
			{
				result.setInteger(0);
			}
		}
		else if ( (left.getType() == TYPE_INTEGER || left.getType() == TYPE_FLOAT) &&
			(right.getType() == TYPE_INTEGER || right.getType() == TYPE_FLOAT))
		{
			if (left.getType() == TYPE_INTEGER)
			{
				if (right.getType() == TYPE_INTEGER)
				{
					if (left.getInteger() == right.getInteger())
					{
						result.setInteger(0);
					}
				}
				else
				{
					if (left.getInteger() == right.getFloat())
					{
						result.setInteger(0);
					}
				}
			}
			else
			{
				if (right.getType() == TYPE_INTEGER)
				{
					if (left.getFloat() == right.getInteger())
					{
						result.setInteger(0);
					}
				}
				else
				{
					if (left.getFloat() == right.getFloat())
					{
						result.setInteger(0);
					}
				}
			}
		}
		else
		{
			sprintf(errmsg, "NEָ����������������Ͳ���ȷ!");
			return -1;
		}
		operation_stack.push(result);
		return 0;
	}
	if (strcmp(inst.inst_action, "NOT") == 0)
	{
		Variable left, result;
		operation_stack.pop(left);
		result.setType(TYPE_INTEGER);
		if (left.getInteger() == 0)
		{
			result.setInteger(1);
		}
		else
		{
			result.setInteger(0);
		}
		operation_stack.push(result);
		return 0;
	}
	if (strcmp(inst.inst_action, "OR") == 0)
	{
		Variable left, right, result;
		operation_stack.pop(right);
		operation_stack.pop(left);
		result.setType(TYPE_INTEGER);
		if (left.getInteger() != 0 || right.getInteger() != 0)
		{
			result.setInteger(1);
		}
		else
		{
			result.setInteger(0);
		}
		operation_stack.push(result);
		return 0;
	}
	if (strcmp(inst.inst_action, "SAVCALL") == 0)
	{
		int index = atoi(inst.inst_operant2 + 1);
		recall_stack.push(index);
		return 0;
	}
	if (strcmp(inst.inst_action, "UMINUS") == 0)
	{
		Variable v, result;
		operation_stack.pop(v);
		if (v.getType() == TYPE_INTEGER)
		{
			result.setType(TYPE_INTEGER);
			result.setInteger(  -(v.getInteger()) );
		}
		else 
		{
			result.setType(TYPE_FLOAT);
			result.setFloat(  -(v.getFloat()) );
		}
		operation_stack.push(result);
		return 0;
	}
	if (strcmp(inst.inst_action, "CLR") == 0)
	{
		operation_stack.removeAll();
		return 0;
	}
	if (strcmp(inst.inst_action, "JMP") == 0)
	{
		Variable v;
		operation_stack.pop(v);
		AnsiString s;
		s = (AnsiString)"F_";
		s.concat(v.getString());
		s.concat("_BEGIN");
		int index = instruct_list.GetIndexByLabel(s.c_str());
		if (index < 0)
		{
			sprintf(errmsg, "JMP��ת��Ŀ�����̿鲻����!");
			return -1;
		}
		instruct_list.JmpToInstruct(index);
		return 0;
	}


	sprintf(errmsg, "����ʶ���ָ��[%s]!\n", inst.inst_action);
	return -1;
}
bool FindVarByName(const AnsiString& name,Variable& ele)
{
	int varnum;
	/*���ݱ������ֺ͵�ǰģ��ı�������������ջ�в��Ҿֲ�����*/
	varnum_stack.peek(varnum);
	if (data_stack.FindVarByNameFrmTop(name, varnum, ele) == TRUE)
	{
		return TRUE;
	}

	/*��ȫ�ֱ���������*/
	int index;
	if (1 == isGlobalID(name.c_str(), &index))
	{
		ele = global_var[index];
		return TRUE;
	}
	return FALSE;	
}
bool ModifyVarByName(const AnsiString &name, const Variable& ele)
{
	/*�Ȳ鿴�ֲ�����ջ*/
	int varnum;
	
	varnum_stack.peek(varnum);
	#ifdef _DEBUG
	/*
		if(name == "mb2")
		{
			const char * p = ele.m_MemBlockValue.GetBufferPtr();
			printf("ModifyVarByName>>>>>>>[%d][%c][%c][%c]\n",ele.m_MemBlockValue.GetSize(), *(p+0),*(p+1),*(p+2));
		}
		*/
	#endif
	if (data_stack.ModifyVarByNameFrmTop(name, varnum, ele) == TRUE)
	{
		return TRUE;
	}
	/*��ȫ�ֱ���������*/
	int index;
	if (1 == isGlobalID(name.c_str(), &index) && ele.getType() == TYPE_MEMBLOCK)
	{

		global_var[index] = ele;
		global_var[index].setName(name);
		return TRUE;
	}
	return FALSE;	
}
