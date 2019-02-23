#include <stdlib.h>
#include <stdio.h>
#include  <string.h>
#include "instruct_list.h"
#include "LabelIndex_List.h"
#include <assert.h>

Instruct_List::Instruct_List()
{
	m_pConstString = NULL;
	m_pInstruct = NULL;
	m_nConstStringCount = 0;
	m_nInstructCount = 0;
	m_nConstStringMax = 0;
	m_nInstructMax = 0;
	m_nStart = 0;
}
Instruct_List::~Instruct_List()
{
	if (NULL != m_pConstString)
	{
		delete[] m_pConstString;
		m_pConstString = NULL;
	}
	if (NULL != m_pInstruct)
	{
		delete[] m_pInstruct;
		m_pInstruct = NULL;
	}
	m_nConstStringCount = 0;
	m_nInstructCount = 0;
	m_nConstStringMax = 0;
	m_nInstructMax = 0;
}
int Instruct_List::Initialize(const char* filename, char * errmsg)
{
	assert(NULL != errmsg);
	assert(NULL != filename);

	/*清除已有的数据*/
	if (NULL != m_pConstString)
	{
		delete[] m_pConstString;
		m_pConstString = NULL;
	}
	if (NULL != m_pInstruct)
	{
		delete[] m_pInstruct;
		m_pInstruct = NULL;
	}
	m_nConstStringCount = 0;
	m_nInstructCount = 0;
	m_nConstStringMax = 0;
	m_nInstructMax = 0;
	m_nStart = 0;

	/*重新分配好缓冲区*/
	m_nConstStringMax = 5;
	m_nInstructMax = 500;

	m_pConstString = new AnsiString[m_nConstStringMax];
	m_pInstruct = new INSTRUCT[m_nInstructMax];
	if (m_pConstString == NULL || m_pInstruct == NULL)
	{
		sprintf(errmsg, "分配内存空间失败!");
		return -1;
	}
	m_nConstStringCount = 0;
	m_nInstructCount = 0;
	

	#define READBUF_MAX 1500

	char buf[READBUF_MAX];

	FILE * fp = NULL;
	if ( (fp = fopen(filename, "rb")) == NULL)
	{
		sprintf(errmsg, "打开文件%s失败!", filename);
		return -1;
	}
	/*读取常量字符串定义部分*/
	while (1)
	{
		if (NULL == fgets(buf, READBUF_MAX, fp))
		{
			sprintf(errmsg, "文件%s不完整!", filename);
			fclose(fp);
			return -1;
		}
		if (strlen(buf) >= (READBUF_MAX - 1))
		{
			sprintf(errmsg, "读取的缓冲区太小或者文件中的行太长!");
			fclose(fp);
			return -1;
		}
		if (memcmp(buf, "%%%%", 2) == 0)
		{
			break;
		}
		/*去掉最后的换行符号*/
		RemoveNL(buf);

		/*格式分解、去除前后引号*/
		char * p = strchr(buf, ' ');	/*空格的位置*/
		if (NULL == p)
		{
			sprintf(errmsg, "常量字符串定义部分的格式不正确!");
			fclose(fp);
			return -1;
		}
		AnsiString astring = (AnsiString)(p + 1);	
		p = NULL;

		memset(buf, 0, READBUF_MAX);
		strcpy(buf, astring.c_str());

		int tmp = strlen(buf);
		while (tmp > 0 && buf[tmp] !='\"')
		{
			tmp--;
		}
		buf[tmp] = 0;
		astring = (AnsiString)(buf + 1);
		/*追加一个元素*/
		if (ReallocConstStringBuf() != 0)
		{
			sprintf(errmsg, "扩展常量字符串列表失败!");
			fclose(fp);
			return -1;
		}
		m_pConstString[m_nConstStringCount++] = astring;
	}
	/*读取指令部分*/
	labelindex_list.removeAll();
	while (1)
	{
		if (NULL == fgets(buf, READBUF_MAX, fp))
		{
			break;
		}

		/*去掉最后的换行符号*/
		RemoveNL(buf);

		if (strcmp(buf, "LABEL F_main_BEGIN") == 0)
		{
			m_nStart = m_nInstructCount;
			#ifdef _DEBUG
			//printf("!!!!m_nStart=[%d]\n", m_nStart);
			#endif
		}

		INSTRUCT inst;
		memset(&inst, 0, sizeof(INSTRUCT));
		int num;
		num = sscanf(buf, "%s %s %s", inst.inst_action, inst.inst_operant1, inst.inst_operant2);
		if (num < 1)
		{
			sprintf(errmsg, "指令%s的格式错误!", buf);
			fclose(fp);
			return -1;
		}
		/*如果是一标签语句，加如到labelIndex_list中*/
		if (strcmp(inst.inst_action, "LABEL") == 0)
		{
			LabelIndex li;
			memset(&li, 0, sizeof(LabelIndex));
			strcpy(li.li_label, inst.inst_operant1);
			li.li_index = m_nInstructCount;
			labelindex_list.AddTail(li);
		}
		/*增加一个新的元素*/
		if (ReallocInstructBuf() != 0)
		{
			sprintf(errmsg, "扩展指令列表失败!");
			fclose(fp);
			return -1;
		}
		m_pInstruct[m_nInstructCount++] = inst;
	}
	fclose(fp);
	/*遍历指令列表，将所有跳转语句的第2操作数赋值为目标标签的索引号*/
	int labelindex_count = labelindex_list.GetSize();
	for (int i = 0; i < m_nInstructCount; i++)
	{
		INSTRUCT inst;
		inst = m_pInstruct[i];
		if (memcmp(inst.inst_action, "GOTO", 4) != 0 && 
			memcmp(inst.inst_action, "SAVCALL", 7) != 0)
		{
			continue;
		}
		int index = labelindex_list.GetIndexByLabel(inst.inst_operant1);
		if (index < 0)
		{
			sprintf(errmsg, "跳转指令非法!");
			return -1;
		}
		sprintf(m_pInstruct[i].inst_operant2, "@%d", index);
	}
	return 0;
	#undef READBUF_MAX 
}
int Instruct_List::JmpToInstruct(int index)
{
	if (index >= m_nInstructCount || index < 0)
	{
		return -1;
	}
	m_nCurrentIndex = index;
	return 0;
}
void Instruct_List::BeginExecute()
{
	m_nCurrentIndex = m_nStart;
}
int Instruct_List::NextInstruct(INSTRUCT & inst)
{
	if (m_nCurrentIndex >= m_nInstructCount)
	{
		return -1;
	}
	inst  = m_pInstruct[m_nCurrentIndex];
	m_nCurrentIndex++;
	return 0;
}
int Instruct_List::GetConstString(int index, AnsiString &s)
{
	if (index < 0 || index >= m_nConstStringCount)
	{
		return -1;
	}
	s = m_pConstString[index];
	return 0;
}
int Instruct_List::ReallocConstStringBuf()
{
	if (m_nConstStringCount < m_nConstStringMax)	/*未满不必扩展缓冲*/
	{
		return 0;
	}
	int max = m_nConstStringMax + 50;	/*一次扩展50个单位*/
	AnsiString * p = new AnsiString[max];
	if (NULL == p)
	{
		return -1;
	}
	/*依次拷贝已有的数据*/
	for (int i = 0; i < m_nConstStringCount; i++)
	{
		p[i] = m_pConstString[i];
	}
	delete[] m_pConstString;
	m_pConstString = p;
	m_nConstStringMax = max;
	return 0;
}
int Instruct_List::ReallocInstructBuf()
{
	if (m_nInstructCount < m_nInstructMax)	/*未满不必扩展缓冲*/
	{
		return 0;
	}
	int max = m_nInstructMax + 500;	/*一次扩展200个单位*/
	INSTRUCT * p = new INSTRUCT[max];
	if (NULL == p)
	{
		return -1;
	}
	/*依次拷贝已有的数据*/
	for (int i = 0; i < m_nInstructCount; i++)
	{
		p[i] = m_pInstruct[i];
	}
	delete[] m_pInstruct;
	m_pInstruct = p;
	m_nInstructMax = max;
	return 0;
}
void Instruct_List::RemoveNL(char * buf)
{
	assert(NULL != buf);
	int i = strlen(buf);
	if (buf[i - 1] == '\n')
	{
		buf[i-1] = 0;
	}
	if (buf[i-2] == '\r')
	{
		buf[i-2] = 0;
	}
}
int Instruct_List::GetIndexByLabel(const char* label)
{
	assert(NULL != label);
	LabelIndex li;
	int index = labelindex_list.GetIndexByLabel(label);
	return index;
}
