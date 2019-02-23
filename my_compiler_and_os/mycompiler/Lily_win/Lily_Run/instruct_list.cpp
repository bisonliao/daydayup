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

	/*������е�����*/
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

	/*���·���û�����*/
	m_nConstStringMax = 5;
	m_nInstructMax = 500;

	m_pConstString = new AnsiString[m_nConstStringMax];
	m_pInstruct = new INSTRUCT[m_nInstructMax];
	if (m_pConstString == NULL || m_pInstruct == NULL)
	{
		sprintf(errmsg, "�����ڴ�ռ�ʧ��!");
		return -1;
	}
	m_nConstStringCount = 0;
	m_nInstructCount = 0;
	

	#define READBUF_MAX 1500

	char buf[READBUF_MAX];

	FILE * fp = NULL;
	if ( (fp = fopen(filename, "rb")) == NULL)
	{
		sprintf(errmsg, "���ļ�%sʧ��!", filename);
		return -1;
	}
	/*��ȡ�����ַ������岿��*/
	while (1)
	{
		if (NULL == fgets(buf, READBUF_MAX, fp))
		{
			sprintf(errmsg, "�ļ�%s������!", filename);
			fclose(fp);
			return -1;
		}
		if (strlen(buf) >= (READBUF_MAX - 1))
		{
			sprintf(errmsg, "��ȡ�Ļ�����̫С�����ļ��е���̫��!");
			fclose(fp);
			return -1;
		}
		if (memcmp(buf, "%%%%", 2) == 0)
		{
			break;
		}
		/*ȥ�����Ļ��з���*/
		RemoveNL(buf);

		/*��ʽ�ֽ⡢ȥ��ǰ������*/
		char * p = strchr(buf, ' ');	/*�ո��λ��*/
		if (NULL == p)
		{
			sprintf(errmsg, "�����ַ������岿�ֵĸ�ʽ����ȷ!");
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
		/*׷��һ��Ԫ��*/
		if (ReallocConstStringBuf() != 0)
		{
			sprintf(errmsg, "��չ�����ַ����б�ʧ��!");
			fclose(fp);
			return -1;
		}
		m_pConstString[m_nConstStringCount++] = astring;
	}
	/*��ȡָ���*/
	labelindex_list.removeAll();
	while (1)
	{
		if (NULL == fgets(buf, READBUF_MAX, fp))
		{
			break;
		}

		/*ȥ�����Ļ��з���*/
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
			sprintf(errmsg, "ָ��%s�ĸ�ʽ����!", buf);
			fclose(fp);
			return -1;
		}
		/*�����һ��ǩ��䣬���絽labelIndex_list��*/
		if (strcmp(inst.inst_action, "LABEL") == 0)
		{
			LabelIndex li;
			memset(&li, 0, sizeof(LabelIndex));
			strcpy(li.li_label, inst.inst_operant1);
			li.li_index = m_nInstructCount;
			labelindex_list.AddTail(li);
		}
		/*����һ���µ�Ԫ��*/
		if (ReallocInstructBuf() != 0)
		{
			sprintf(errmsg, "��չָ���б�ʧ��!");
			fclose(fp);
			return -1;
		}
		m_pInstruct[m_nInstructCount++] = inst;
	}
	fclose(fp);
	/*����ָ���б���������ת���ĵ�2��������ֵΪĿ���ǩ��������*/
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
			sprintf(errmsg, "��תָ��Ƿ�!");
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
	if (m_nConstStringCount < m_nConstStringMax)	/*δ��������չ����*/
	{
		return 0;
	}
	int max = m_nConstStringMax + 50;	/*һ����չ50����λ*/
	AnsiString * p = new AnsiString[max];
	if (NULL == p)
	{
		return -1;
	}
	/*���ο������е�����*/
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
	if (m_nInstructCount < m_nInstructMax)	/*δ��������չ����*/
	{
		return 0;
	}
	int max = m_nInstructMax + 500;	/*һ����չ200����λ*/
	INSTRUCT * p = new INSTRUCT[max];
	if (NULL == p)
	{
		return -1;
	}
	/*���ο������е�����*/
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
