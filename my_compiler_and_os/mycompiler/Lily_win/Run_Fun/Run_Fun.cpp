#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Variable.h"
#include "AnsiString.h"
#include "Run_Fun.h" 

#undef SHOW_STEP
#if defined(_DEBUG)
	#undef SHOW_STEP
	#define SHOW_STEP printf("ִ��%s�ĵ�%d��...\n", __FILE__, __LINE__);
	//#define SHOW_STEP ;
#else
	#define SHOW_STEP ;
#endif

void printHex(const char * buf, int len);
//////////////////////////////////////////////////////////////

int print_proc(int argc, Variable*argv);
int atof_proc(int argc, Variable* argv);
int atoi_proc(int argc, Variable* argv);
int str_proc(int argc, Variable*argv);
int substr_proc(int argc, Variable* argv);
int system_proc(int argc, Variable* argv);
int malloc_proc(int argc, Variable* argv);
int memsub_proc(int argc, Variable* argv);
int memadd_proc(int argc, Variable* argv);
int mem2str_proc(int argc, Variable* argv);
int str2mem_proc(int argc, Variable* argv);
int memcmp_proc(int argc, Variable* argv);
int length_proc(int argc, Variable* argv);

//////////////////////
//�ܵ����
extern "C"  int Run_Fun(int funnum, int argc, Variable* argv)
{
	switch (funnum)
	{
		case 1001:
			return print_proc(argc, argv);
		case 1002:
			return str_proc(argc, argv);
		case 1003:
			return atoi_proc(argc, argv);
		case 1004:
			return atof_proc(argc, argv);
		case 1005:
			return substr_proc(argc, argv);
		case 1006:
			return system_proc(argc, argv);
		case 1007:
			return malloc_proc(argc, argv);
		case 1008:
			return memsub_proc(argc, argv);
		case 1009:
			return memadd_proc(argc, argv);
		case 1010:
			return mem2str_proc(argc, argv);
		case 1011:
			return str2mem_proc(argc, argv);
		case 1012:
			return memcmp_proc(argc, argv);
		case 1013:
			return length_proc(argc, argv);
		default:
			break;
	}
	return -1;
}
//�ڱ�׼�����ӡһ�����
int print_proc(int argc, Variable*argv)
{
	int i;

	for (i = 1; i <=argc; i++)
	{
		Variable * vp = argv + i;
		printf("[%s]", vp->getName().c_str());
		switch (vp->getType())
		{
			case TYPE_STRING:
				printf("[string][%s]\n", vp->getString().c_str());
				break;
			case TYPE_INTEGER:
				printf("[int][%d]\n", vp->getInteger());
				break;
			case TYPE_FLOAT:
				printf("[float][%f]\n", vp->getFloat());
				break;
			case TYPE_MEMBLOCK:
				printf("[memblock]");
				printHex(vp->m_MemBlockValue.GetBufferPtr(),
					vp->m_MemBlockValue.GetSize());
				
				break;
			default:
				break;
		}
	}	
	argv[0].setType(TYPE_INTEGER);
	argv[0].setInteger(0);
	return 0;
}
//��������������ת��Ϊ�ַ���
int str_proc(int argc, Variable*argv)
{
	if (argc != 1)
	{
		return -1;
	}
	argv[0].setType(TYPE_STRING);
	argv[0].setString((AnsiString)"��Ч����");
	char tmp[100];
	if (argv[1].getType() == TYPE_INTEGER)
	{
		sprintf(tmp, "%d", argv[1].getInteger());
		argv[0].setString((AnsiString)tmp);
	}
	else if (argv[1].getType() == TYPE_FLOAT)
	{
		sprintf(tmp, "%f", argv[1].getFloat());
		argv[0].setString((AnsiString)tmp);
	}
	else if (argv[1].getType() == TYPE_STRING)
	{
		argv[0].setString(argv[1].getString());
	}
	else
	{
		return 0;
	}
	return 0;
}
//���ַ���ת��Ϊ����
int atoi_proc(int argc, Variable* argv)
{	
	if (argc != 1)
	{
		return -1;
	}
	argv[0].setType(TYPE_INTEGER);
	argv[0].setInteger(0);
	if (argv[1].getType() == TYPE_STRING)
	{
		argv[0].setInteger(atoi(argv[1].getString().c_str()));
	}
	else
	{
		return 0;
	}
	return 0;
}
//���ַ���ת��Ϊ������
int atof_proc(int argc, Variable* argv)
{	
	if (argc != 1)
	{
		return -1;
	}
	argv[0].setType(TYPE_FLOAT);
	argv[0].setFloat(0);
	if (argv[1].getType() == TYPE_STRING)
	{
		argv[0].setFloat(atof(argv[1].getString().c_str()));
	}
	else
	{
		return 0;
	}
	return 0;
}
//�õ�һ���ַ������Ӵ�
//substr(string src, int offset, int len)
int substr_proc(int argc, Variable* argv)
{	
	argv[0].setType(TYPE_STRING);
	argv[0].setString("");
	if (argc != 3)
	{
		return -1;
	}
	if (argv[1].getType() != TYPE_STRING || 
		argv[2].getType() != TYPE_INTEGER ||
		argv[3].getType() != TYPE_INTEGER)
	{
		return 0;
	}
	
	argv[0].setString(
		argv[1].m_StringValue.substring(
			argv[2].getInteger(), argv[3].getInteger())
			);
	return 0;
}
//����һ���µ��ӽ���ִ��һ��shell����
int system_proc(int argc, Variable* argv)
{
	argv[0].setType(TYPE_INTEGER);
	argv[0].setInteger(-1);
	if (argv[1].getType() != TYPE_STRING)
	{
		return 0;
	}
	argv[0].setInteger(system(argv[1].getString().c_str()));
	return 0;
}
int malloc_proc(int argc, Variable* argv)
{
	// MemBlock malloc(Integer val, Integer size);
	// ����һ���СΪsize���ڴ�����ÿ�ֽ�ֵ����Ϊval
	
	
	argv[0].setType(TYPE_MEMBLOCK);
	
	//�����������ȷ
	if (argv[1].getType() != TYPE_INTEGER ||
		argv[2].getType() != TYPE_INTEGER)
	{
		return 0;
	}
	if (argv[2].getInteger() < 0)
	{
		return 0;
	}
	argv[0].m_MemBlockValue.Realloc(argv[2].getInteger());
	argv[0].m_MemBlockValue.MemSet(argv[1].getInteger(), argv[2].getInteger());
	return 0;
}
int memsub_proc(int argc, Variable* argv)
{
	// MemBlock memsub(MemBlock src, Integer offset, Integer len);
	// ���ش�srcƫ��Ϊoffset����Ϊsize��һ���ڴ�
	
	argv[0].setType(TYPE_MEMBLOCK);
	
	//printf(">>>>>>>>memsub_proc: %s", argv[0].toString().c_str());
	if (argv[1].getType() != TYPE_MEMBLOCK ||
		argv[2].getType() != TYPE_INTEGER ||
		argv[3].getType() != TYPE_INTEGER)
	{
		return 0;
	}
	
	argv[0].m_MemBlockValue = argv[1].m_MemBlockValue.MemSub(argv[2].getInteger(), argv[3].getInteger());
	//printf(">>>>>>>>memsub_proc: %s", argv[0].toString().c_str());	
	//printHex(argv[0].m_MemBlockValue.GetBufferPtr(), argv[0].m_MemBlockValue.GetSize());
	return 0;
}
int memadd_proc(int argc, Variable* argv)
{
	// MemBlock memadd(MemBlock src1, MemBlock src2);
	// �����ڴ����
	
	argv[0].setType(TYPE_MEMBLOCK);//��������ֵ����Ϊ MemBlock
	
	if (argv[1].getType() != TYPE_MEMBLOCK ||
		argv[2].getType() != TYPE_MEMBLOCK)
	{
		return 0;
	}
	argv[0].m_MemBlockValue = argv[1].m_MemBlockValue;
	argv[0].m_MemBlockValue.Append(argv[2].m_MemBlockValue.GetBufferPtr(), argv[2].m_MemBlockValue.GetSize());
	return 0;
}
int mem2str_proc(int argc, Variable* argv)
{
	// string mem2str(MemBlock src);
	// ��һ���ڴ��ת��Ϊ�ַ���
	argv[0].setType(TYPE_STRING);//��������ֵ����Ϊ string
	
	
	if (argv[1].getType() != TYPE_MEMBLOCK)
	{
		return 0;
	}

	char nl[1];
	nl[0] = 0;
	argv[1].m_MemBlockValue.Append(nl, 1);
	argv[0].m_StringValue = AnsiString(argv[1].m_MemBlockValue.GetBufferPtr());
	argv[0].m_StringValue.trimToSize();
	
	return 0;
}
int str2mem_proc(int argc, Variable* argv)
{
	// string mem2str(MemBlock src);
	// ��һ���ڴ��ת��Ϊ�ַ���
	
	argv[0].setType(TYPE_MEMBLOCK);//��������ֵ����Ϊ MemBlock
	
	if (argv[1].getType() != TYPE_STRING)
	{
		return 0;
	}
	argv[0].m_MemBlockValue.SetValue(argv[1].m_StringValue.c_str(), argv[1].m_StringValue.length()+1);
	return 0;
}
int memcmp_proc(int argc, Variable* argv)
{
	// int mem2str(MemBlock mb1, MemBlock mb2);
	//�Ƚ������ڴ���Ƿ����,��ȷ���1�� ����ȷ���0
	
	argv[0].setType(TYPE_INTEGER);//��������ֵ����Ϊ int
	argv[0].setInteger(0);
	
	if (argv[1].getType() != TYPE_MEMBLOCK ||
		argv[2].getType() != TYPE_MEMBLOCK)
	{
		return 0;
	}
	if (argv[1].m_MemBlockValue.GetSize() != argv[2].m_MemBlockValue.GetSize())
	{
		return 0;
	}
	if (memcmp(argv[1].m_MemBlockValue.GetBufferPtr(),
		argv[2].m_MemBlockValue.GetBufferPtr(),
		argv[1].m_MemBlockValue.GetSize()) == 0)
	{
		argv[0].setInteger(1);
		return 0;
	}
	return 0;
}
int length_proc(int argc, Variable* argv)
{
	//����һ���ڴ�����ַ����ĳ���
	
	argv[0].setType(TYPE_INTEGER);//��������ֵ����Ϊ int
	argv[0].setInteger(0);
	if (argv[1].getType() == TYPE_STRING)
	{
		argv[0].setInteger(argv[1].m_StringValue.length());
	}
	else if (argv[1].getType() == TYPE_MEMBLOCK)
	{
		argv[0].setInteger(argv[1].m_MemBlockValue.GetSize());
	}
	
	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////
void printHex(const char * buf, int len)
{
	int i, j;
	int line;
	int start;
	int end;
	if (NULL == buf || len <= 0)
	{
		return;
	}
	
	line = len / 10;
	if (len % 10)
	{
		line++;
	}
	fprintf(stdout, "\n");
	for (i = 0; i < line; i++)
	{
		start = i * 10;
		end = start + 10;
		fprintf(stdout, "%04d:", start);
		for (j = start; j < end && j < len; j++)
		{
			fprintf(stdout, "%2X ", buf[j]);
		}
		for (; j < end ;j++)
		{
			fprintf(stdout, "   ");
		}
		fprintf(stdout, "||");
		for (j = start; j < end && j < len; j++)
		{
			if (buf[j] == 0 || buf[j] == '\r' || buf[j] == '\n')
			{
				fprintf(stdout, ". ");
			}
			else
			{
				fprintf(stdout, "%c ", buf[j]);
			}
		}
		fprintf(stdout, "\n");
	}

	return;
}