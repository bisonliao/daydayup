#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Variable.h"
#include "AnsiString.h"

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
int gets_proc(int argc, Variable* argv);

extern "C" int Run_Fun(int funnum, int argc, Variable* argv)
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
		case 1014:
			return gets_proc(argc, argv);
		default:
			break;
	}
	return -1;
}
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
				printf("[memblock][内存块不被打印]\n");
				break;
			default:
				break;
		}
	}	
	argv[0].setType(TYPE_INTEGER);
	argv[0].setInteger(0);
	return 0;
}
int str_proc(int argc, Variable*argv)
{
	if (argc != 1)
	{
		return -1;
	}
	argv[0].setType(TYPE_STRING);
	argv[0].setString((AnsiString)"无效数据");
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
		return -1;
	}
	return 0;
}
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
		return -1;
	}
	return 0;
}
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
		return -1;
	}
	return 0;
}
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
		return -1;
	}
	int len = argv[1].getString().length();
	if (argv[2].getInteger() < 0 || argv[3].getInteger() < 0)
	{
		return -1;
	}
	argv[0].setType(TYPE_STRING);
	AnsiString s = argv[1].getString().substring( argv[2].getInteger(), 
			argv[3].getInteger());
	argv[0].setString(s);
	return 0;
}
int system_proc(int argc, Variable* argv)
{
	argv[0].setType(TYPE_INTEGER);
	argv[0].setInteger(0);
	if (argv[1].getType() != TYPE_STRING)
	{
		return -1;
	}
	argv[0].setInteger(system(argv[1].getString().c_str()));
	return 0;
}
int malloc_proc(int argc, Variable* argv)
{
	// MemBlock malloc(Integer val, Integer size);
	// 返回一块大小为size的内存区，每字节值设置为val
	
	
	argv[0].setType(TYPE_MEMBLOCK);
	
	//传入参数不正确
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
	// 返回从src偏移为offset长度为size的一段内存
	
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
	// 两段内存相加
	
	argv[0].setType(TYPE_MEMBLOCK);//函数返回值类型为 MemBlock
	
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
	// 将一个内存块转化为字符串
	argv[0].setType(TYPE_STRING);//函数返回值类型为 string
	
	
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
	// 将一个内存块转化为字符串
	
	argv[0].setType(TYPE_MEMBLOCK);//函数返回值类型为 MemBlock
	
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
	//比较两个内存块是否相等,相等返回1， 不相等返回0
	
	argv[0].setType(TYPE_INTEGER);//函数返回值类型为 int
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
	//返回一段内存或者字符串的长度
	
	argv[0].setType(TYPE_INTEGER);//函数返回值类型为 int
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
int gets_proc(int argc, Variable* argv)
{
	argv[0].setType(TYPE_INTEGER);
	argv[0].setInteger(0);
	if (argv[1].getType() != TYPE_STRING)
	{
		fprintf(stderr, "gets需要一个字符串的地址作为参数\n");
		return -1;
	}
	//将保存在字符串argv[1]里的16进制表示的变量地址取出来
	char addr[100];
	memset(addr, 0, sizeof(addr));
	strncpy(addr, argv[1].getString().c_str(), sizeof(addr));
	unsigned long point;
	if (sscanf(addr, "%x", &point) != 1)
	{
		fprintf(stderr, "从字符串[%s]中取地址失败\n", addr);
		return -2;
	}
	Variable * p = (Variable*)point;
	if (p->getType() != TYPE_STRING)
	{
		fprintf(stderr, "gets需要一个 *字符串* 的地址作为参数\n");
		return -3;
	}
	char buf[255];
	memset(buf, 0, sizeof(buf));
	fgets(buf, sizeof(buf), stdin);
	p->setString(buf);
	return 0;
}
