#if !defined(_WIN32_)
#include <unistd.h>
#endif
#include <stdlib.h>
//#include <iostream>
#include <stdio.h>
#include <fstream.h>
#include <string.h>
#include "lex1/FlexLexer.h"

#include "xml.h"
#include "xmllex.h"

#if defined(_WIN32_)
	#include <io.h>

	#define snprintf _snprintf
	#define open	_open
	#define dup2	_dup2
#endif

#include <stack>

#ifdef _DEBUG
	#define SHOW_STEP {printf("���е�[%s][%d]\n", __FILE__, __LINE__);}
#endif


Xml::Xml()
{
	m_RootPtr = NULL;
}
Xml::~Xml()
{
	Clear();
}

typedef struct 
{
	XmlNode * NodePtr;
	list<XmlNode*>::const_iterator It;
} DEPTH_SCAN_STACK_ELE_TYPE;
/*
*	������������������ȵķǵݹ�������õ����нڵ������
*	������lst�У��ɹ�����0��ʧ�ܷ���-1
*/
int Xml::DepthScan(list<XmlNode*> & lst) const
{
	lst.clear();

	if (m_RootPtr == NULL)
	{
		return 0;
	}

	stack<DEPTH_SCAN_STACK_ELE_TYPE> stk;
	DEPTH_SCAN_STACK_ELE_TYPE stk_ele, stk_ele2;
	list<XmlNode*>::const_iterator it;
	XmlNode * curptr = NULL;

	stk_ele.NodePtr = m_RootPtr;
	stk_ele.It = m_RootPtr->GetNodeChildList().begin();
	stk.push(stk_ele);


	while (!stk.empty())
	{
		stk_ele2 = stk.top();
		if (stk_ele2.NodePtr->GetNodeChildList().empty())	/*Ҷ�ӽڵ�*/
		{
			lst.push_back(stk_ele2.NodePtr);
			stk.pop();
		}
		else	/*����Ҷ��*/
		{
			it = stk_ele2.It;
			if (it == stk_ele2.NodePtr->GetNodeChildList().end()) /*���к��Ӷ�������*/
			{
				lst.push_back(stk_ele2.NodePtr);
				stk.pop();
			}
			else /*������һ������*/
			{
				//ȡ���ӽڵ�ָ�롢��ʼ���ӽڵ�ĵ�����
				stk_ele.NodePtr = *it;
				stk_ele.It = stk_ele.NodePtr->GetNodeChildList().begin();
				//�޸�ջ���ڵ��It��ֵ
				stk.pop();
				++(stk_ele2.It);
				stk.push(stk_ele2);
				//�����С��ѹջ
				stk.push(stk_ele);
			}
		}
	}
	return 0;
}
#ifdef OLD_20050124
int Xml::WriteToBuffer(char * buffer, unsigned short buffersize)
{
	if (buffersize == 0)
	{
		return -1;
	}
	if (m_RootPtr == NULL)
	{
		strcpy(buffer, "");
		return 0;
	}

	stack<DEPTH_SCAN_STACK_ELE_TYPE> stk;
	DEPTH_SCAN_STACK_ELE_TYPE stk_ele, stk_ele2;
	list<XmlNode*>::const_iterator it;
	XmlNode * curptr = NULL;

	stk_ele.NodePtr = m_RootPtr;
	stk_ele.It = m_RootPtr->GetNodeChildList().begin();
	stk.push(stk_ele);

	/*д���ڵ�Ŀ�ʼ���ź�����*/
	AnsiString xmlstring =  "<?xml version=\"1.0\" encoding=\"GB2312\" standalone='no'?>\n";
	xmlstring.concat("<");
	xmlstring.concat(stk_ele.NodePtr->GetNodeName());
	if (stk_ele.NodePtr->GetNodeProperty().length() == 0)
	{
		xmlstring.concat(">");
	}
	else
	{
		xmlstring.concat(" ");
		xmlstring.concat(stk_ele.NodePtr->GetNodeProperty());
		xmlstring.concat(">");
	}

	while (!stk.empty())
	{
		stk_ele2 = stk.top();
		if (stk_ele2.NodePtr->GetNodeChildList().empty())	/*Ҷ�ӽڵ�*/
		{
			/*дҶ�ӽڵ��ֵ*/
			xmlstring.concat(stk_ele2.NodePtr->GetNodeValue());
			/*дҶ�ӽڵ�Ľ�������*/
			xmlstring.concat("</");
			xmlstring.concat(stk_ele2.NodePtr->GetNodeName());
			xmlstring.concat(">\n");

			stk.pop();
		}
		else	/*����Ҷ��*/
		{
			it = stk_ele2.It;
			if (it == stk_ele2.NodePtr->GetNodeChildList().end()) /*���к��Ӷ�������*/
			{
				xmlstring.concat("</");
				xmlstring.concat(stk_ele2.NodePtr->GetNodeName());
				xmlstring.concat(">");
				stk.pop();
			}
			else /*������һ������*/
			{
				//ȡ���ӽڵ�ָ�롢��ʼ���ӽڵ�ĵ�����
				stk_ele.NodePtr = *it;
				stk_ele.It = stk_ele.NodePtr->GetNodeChildList().begin();
				//�޸�ջ���ڵ��It��ֵ
				stk.pop();
				++(stk_ele2.It);
				stk.push(stk_ele2);
				//�����С��ѹջ
				xmlstring.concat("<");
				xmlstring.concat(stk_ele.NodePtr->GetNodeName());
				if (stk_ele.NodePtr->GetNodeProperty().length() == 0)
				{
					xmlstring.concat(">");
				}
				else
				{
					xmlstring.concat(" ");
					xmlstring.concat(stk_ele.NodePtr->GetNodeProperty());
					xmlstring.concat(">");
				}
				stk.push(stk_ele);
			}
		}
	}
	strncpy(buffer, xmlstring.c_str(), buffersize);
	return 0;
}

int Xml::WriteToFile(const char * xmlfilename, char* errmsg, unsigned short errmsgsize)
{
	FILE * out;
	if ( (out = fopen(xmlfilename, "wb+")) == NULL)
	{
	
		snprintf(errmsg, errmsgsize, "���ļ�[%s]ʧ��!", xmlfilename);
	
		return -1;
	}
	if (m_RootPtr == NULL)
	{
		fclose(out);
		return 0;
	}

	stack<DEPTH_SCAN_STACK_ELE_TYPE> stk;
	DEPTH_SCAN_STACK_ELE_TYPE stk_ele, stk_ele2;
	list<XmlNode*>::const_iterator it;
	XmlNode * curptr = NULL;

	stk_ele.NodePtr = m_RootPtr;
	stk_ele.It = m_RootPtr->GetNodeChildList().begin();
	stk.push(stk_ele);

	/*д���ڵ�Ŀ�ʼ���ź�����*/
	fprintf(out, "<?xml version=\"1.0\" encoding=\"GB2312\" standalone='no'?>\n");
	fprintf(out, "<%s", stk_ele.NodePtr->GetNodeName().c_str());
	if (stk_ele.NodePtr->GetNodeProperty().length() == 0)
	{
		fprintf(out, ">");
	}
	else
	{
		fprintf(out, " %s>", stk_ele.NodePtr->GetNodeProperty().c_str());
	}

	while (!stk.empty())
	{
		stk_ele2 = stk.top();

		if (stk_ele2.NodePtr->GetNodeChildList().empty())	/*Ҷ�ӽڵ�*/
		{
			/*дҶ�ӽڵ��ֵ*/
			fprintf(out, "%s", stk_ele2.NodePtr->GetNodeValue().c_str());
			/*дҶ�ӽڵ�Ľ�������*/
			fprintf(out, "</%s>\n", stk_ele2.NodePtr->GetNodeName().c_str());

			stk.pop();
		}
		else	/*����Ҷ��*/
		{
			it = stk_ele2.It;
			if (it == stk_ele2.NodePtr->GetNodeChildList().end()) /*���к��Ӷ�������*/
			{
				fprintf(out, "</%s>", stk_ele2.NodePtr->GetNodeName().c_str());
				stk.pop();
			}
			else /*������һ������*/
			{
				//ȡ���ӽڵ�ָ�롢��ʼ���ӽڵ�ĵ�����
				stk_ele.NodePtr = *it;
				stk_ele.It = stk_ele.NodePtr->GetNodeChildList().begin();
				//�޸�ջ���ڵ��It��ֵ
				stk.pop();
				++(stk_ele2.It);
				stk.push(stk_ele2);
				//�����С��ѹջ
				fprintf(out, "<%s", stk_ele.NodePtr->GetNodeName().c_str());
				if (stk_ele.NodePtr->GetNodeProperty().length() == 0)
				{
					fprintf(out, ">");
				}
				else
				{
					fprintf(out, " %s>", stk_ele.NodePtr->GetNodeProperty().c_str());
				}
				stk.push(stk_ele);
			}
		}
	}
	fclose(out);
	return 0;
}
#endif

int Xml::WriteToBuffer(char * buffer, unsigned short buffersize)
{
	if (buffersize == 0)
	{
		return -1;
	}
	if (m_RootPtr == NULL)
	{
		strcpy(buffer, "");
		return 0;
	}

	stack<DEPTH_SCAN_STACK_ELE_TYPE> stk;
	DEPTH_SCAN_STACK_ELE_TYPE stk_ele, stk_ele2;
	list<XmlNode*>::const_iterator it;
	XmlNode * curptr = NULL;

	stk_ele.NodePtr = m_RootPtr;
	stk_ele.It = m_RootPtr->GetNodeChildList().begin();
	stk.push(stk_ele);

	/*д���ڵ�Ŀ�ʼ���ź�����*/
	AnsiString xmlstring =  "<?xml version=\"1.0\" encoding=\"GB2312\" standalone='no'?>\n";
	xmlstring.concat("<");
	xmlstring.concat(stk_ele.NodePtr->GetNodeName());
	if (stk_ele.NodePtr->GetNodeProperty().length() == 0)
	{
		xmlstring.concat(">");
	}
	else
	{
		xmlstring.concat(" ");
		xmlstring.concat(stk_ele.NodePtr->GetNodeProperty());
		xmlstring.concat(">");
	}

	while (!stk.empty())
	{
		stk_ele2 = stk.top();
		if (stk_ele2.NodePtr->GetNodeChildList().empty())	/*Ҷ�ӽڵ�*/
		{
			if (stk_ele2.NodePtr->GetNodeValue().length() > 0)
			{
				xmlstring.concat(">");
				/*дҶ�ӽڵ��ֵ*/
				xmlstring.concat(stk_ele2.NodePtr->GetNodeValue());
				/*дҶ�ӽڵ�Ľ�������*/
				xmlstring.concat("</");
				xmlstring.concat(stk_ele2.NodePtr->GetNodeName());
				xmlstring.concat(">\n");
			}
			else
			{
				xmlstring.concat("/>\n");
			}

			stk.pop();
		}
		else	/*����Ҷ��*/
		{
			it = stk_ele2.It;
			if (it == stk_ele2.NodePtr->GetNodeChildList().end()) /*���к��Ӷ�������*/
			{
				xmlstring.concat("</");
				xmlstring.concat(stk_ele2.NodePtr->GetNodeName());
				xmlstring.concat(">");
				stk.pop();
			}
			else /*������һ������*/
			{
				//ȡ���ӽڵ�ָ�롢��ʼ���ӽڵ�ĵ�����
				stk_ele.NodePtr = *it;
				stk_ele.It = stk_ele.NodePtr->GetNodeChildList().begin();
				//�޸�ջ���ڵ��It��ֵ
				stk.pop();
				++(stk_ele2.It);
				stk.push(stk_ele2);
				//�����С��ѹջ
				if (stk_ele.NodePtr->GetNodeChildList().empty())//С����Ҷ�ӽڵ�
				{
					xmlstring.concat("<");
					xmlstring.concat(stk_ele.NodePtr->GetNodeName());
					if (stk_ele.NodePtr->GetNodeProperty().length() != 0)
					{
						xmlstring.concat(" ");
						xmlstring.concat(stk_ele.NodePtr->GetNodeProperty());
					}
				}
				else
				{
					xmlstring.concat("<");
					xmlstring.concat(stk_ele.NodePtr->GetNodeName());
					if (stk_ele.NodePtr->GetNodeProperty().length() == 0)
					{
						xmlstring.concat(">");
					}
					else
					{
						xmlstring.concat(" ");
						xmlstring.concat(stk_ele.NodePtr->GetNodeProperty());
						xmlstring.concat(">");
					}
				}
				stk.push(stk_ele);
			}
		}
	}
	strncpy(buffer, xmlstring.c_str(), buffersize);
	return 0;
}

int Xml::WriteToFile(const char * xmlfilename, char* errmsg, unsigned short errmsgsize)
{
	FILE * out;
	if ( (out = fopen(xmlfilename, "wb+")) == NULL)
	{
	
		snprintf(errmsg, errmsgsize, "���ļ�[%s]ʧ��!", xmlfilename);
	
		return -1;
	}
	if (m_RootPtr == NULL)
	{
		fclose(out);
		return 0;
	}

	stack<DEPTH_SCAN_STACK_ELE_TYPE> stk;
	DEPTH_SCAN_STACK_ELE_TYPE stk_ele, stk_ele2;
	list<XmlNode*>::const_iterator it;
	XmlNode * curptr = NULL;

	stk_ele.NodePtr = m_RootPtr;
	stk_ele.It = m_RootPtr->GetNodeChildList().begin();
	stk.push(stk_ele);

	/*д���ڵ�Ŀ�ʼ���ź�����*/
	fprintf(out, "<?xml version=\"1.0\" encoding=\"GB2312\" standalone='no'?>\n");
	fprintf(out, "<%s", stk_ele.NodePtr->GetNodeName().c_str());
	if (stk_ele.NodePtr->GetNodeProperty().length() == 0)
	{
		fprintf(out, ">");
	}
	else
	{
		fprintf(out, " %s>", stk_ele.NodePtr->GetNodeProperty().c_str());
	}

	while (!stk.empty())
	{
		stk_ele2 = stk.top();

		if (stk_ele2.NodePtr->GetNodeChildList().empty())	/*Ҷ�ӽڵ�*/
		{
			if (stk_ele2.NodePtr->GetNodeValue().length() > 0)
			{
				fprintf(out, ">");
				/*дҶ�ӽڵ��ֵ*/
				fprintf(out, "%s", stk_ele2.NodePtr->GetNodeValue().c_str());
				/*дҶ�ӽڵ�Ľ�������*/
				fprintf(out, "</%s>\n", stk_ele2.NodePtr->GetNodeName().c_str());
			}
			else
			{
				fprintf(out, "/>\n");
			}

			stk.pop();
		}
		else	/*����Ҷ��*/
		{
			it = stk_ele2.It;
			if (it == stk_ele2.NodePtr->GetNodeChildList().end()) /*���к��Ӷ�������*/
			{
				fprintf(out, "</%s>", stk_ele2.NodePtr->GetNodeName().c_str());
				stk.pop();
			}
			else /*������һ������*/
			{
				//ȡ���ӽڵ�ָ�롢��ʼ���ӽڵ�ĵ�����
				stk_ele.NodePtr = *it;
				stk_ele.It = stk_ele.NodePtr->GetNodeChildList().begin();
				//�޸�ջ���ڵ��It��ֵ
				stk.pop();
				++(stk_ele2.It);
				stk.push(stk_ele2);
				//�����С��ѹջ
				if (stk_ele.NodePtr->GetNodeChildList().empty())//С����Ҷ�ӽڵ�
				{
					fprintf(out, "<%s", stk_ele.NodePtr->GetNodeName().c_str());
					if (stk_ele.NodePtr->GetNodeProperty().length() != 0)
					{
						fprintf(out, " %s", stk_ele.NodePtr->GetNodeProperty().c_str());
					}
				}
				else
				{
					fprintf(out, "<%s", stk_ele.NodePtr->GetNodeName().c_str());
					if (stk_ele.NodePtr->GetNodeProperty().length() == 0)
					{
						fprintf(out, ">");
					}
					else
					{
						fprintf(out, " %s>", stk_ele.NodePtr->GetNodeProperty().c_str());
					}
				}
				stk.push(stk_ele);
			}
		}
	}
	fclose(out);
	return 0;
}
int Xml::ReadFrmFile(const char * xmlfilename, char *errmsg, unsigned short errmsgsize)
{
	this->Clear();
	m_RootPtr = NULL;
	XmlNode* CurPtr = NULL;
	XmlNode* newptr = NULL;
	int nCode = -1;


#ifdef _WIN32_
	::ifstream in(xmlfilename, ios::in|ios::binary);
#else
	ifstream in(xmlfilename, ios::in|ios::binary);
#endif
	yyFlexLexer lexer(&in);
	lexer.g_lineno = 1;

	int retcode;
	int nLevel = 0; /*��Σ����ڵ�Ϊ��1��*/

	stack<XmlNode*> stk;
	stk.push(NULL);


    while ( (retcode = lexer.yylex()) != 0)
	{
		/*ȡ�õ�������*/
		const YYVAL_TYPE *yylval = lexer.__GetYYVal();


		switch(retcode)
		{
			/*
			case XML_LEX_HEADER:
					if (nLevel != 0)
					{
		
						snprintf(errmsg, errmsgsize, "[%d]�ļ�%s�д���!unexpected <?XML...?>",
									__LINE__,
									xmlfilename);
				
						nCode=-1;
						goto ReadFrmFile_EXIT;
					}
					break;
			*/
			case XML_LEX_BEGIN:
					++nLevel;
					/*�½�һ�ڵ�*/
					newptr = new XmlNode();
					if (newptr == NULL)
					{
						snprintf(errmsg, errmsgsize, "�ڴ����ʧ��!");
						nCode=-1;
						goto ReadFrmFile_EXIT;
					}
					if (nLevel == 1)
					{
						m_RootPtr = newptr;
					}
					newptr->SetNodeName(yylval->name);	
					newptr->SetNodeProperty(yylval->property);
					newptr->SetNodeValue("");	
					newptr->SetParent(CurPtr);

					if (CurPtr != NULL)
					{
						CurPtr->SetNodeValue("");	/*�������ӽڵ�Ľڵ㣬��ֵΪ��*/
						CurPtr->AddChild(newptr);
					}
					stk.push(newptr);
					/*���µ�ǰ�ڵ�ָ��*/
					CurPtr = newptr;

					break;
			case XML_LEX_EMPNODE:
					++nLevel;
					/*�½�һ�ڵ�*/
					newptr = new XmlNode();
					if (newptr == NULL)
					{
						snprintf(errmsg, errmsgsize, "�ڴ����ʧ��!");
						nCode=-1;
						goto ReadFrmFile_EXIT;
					}
					if (nLevel == 1)
					{
						m_RootPtr = newptr;
					}
					newptr->SetNodeName(yylval->name);	
					newptr->SetNodeProperty(yylval->property);
					newptr->SetNodeValue("");	
					newptr->SetParent(CurPtr);

					if (CurPtr != NULL)
					{
						CurPtr->SetNodeValue("");	/*�������ӽڵ�Ľڵ㣬��ֵΪ��*/
						CurPtr->AddChild(newptr);
					}

					nLevel--;
					if (nLevel == 0)
					{
						return 0;
					}
					break;
			case XML_LEX_END:
					if ( !(CurPtr->GetNodeName()==AnsiString(yylval->name)) )
					{
						snprintf(errmsg, errmsgsize, 
						"[%d]�ļ�%s�д���!�ֶ�����ǰ��ƥ��[%s][%s]",
								__LINE__,
								xmlfilename,
								CurPtr->GetNodeName().c_str(),
								yylval->name);
						nCode=-1;
						goto ReadFrmFile_EXIT;
					}

					--nLevel;
					if (nLevel < 0 || stk.size() == 0)
					{
						snprintf(errmsg, errmsgsize, "[%d]�ļ�%s�д���!�ڵ㿪�շ��Ų�ƥ��",
								__LINE__,
								xmlfilename);
						nCode=-1;
						goto ReadFrmFile_EXIT;
					}
					/*���µ�ǰ�ڵ�ָ��*/
					stk.pop();
					CurPtr = stk.top();
					if (nLevel == 0)
					{
						return 0;
					}

					break;
			case XML_LEX_VALUE:
					if (CurPtr != NULL)
					{
						CurPtr->SetNodeValue(yylval->value);
					}
					else
					{
						snprintf(errmsg, errmsgsize, "[%d]�ļ�%s�д���! nlevel=[%d]",
									__LINE__,
									xmlfilename, nLevel);
						nCode=-1;
						goto ReadFrmFile_EXIT;
					}
					break;
			default:
					snprintf(errmsg, errmsgsize, "[%d]�ļ�%s�д���!unexpected character[%d][%c]",
								__LINE__,
								xmlfilename,
								retcode,
								retcode);
					nCode=-1;
					goto ReadFrmFile_EXIT;
					break;
		}
	}
	if (nLevel != 0)
	{
		snprintf(errmsg, errmsgsize, "�ļ�%s��Xml�ڵ�Ŀ�ʼ���źͽ������Ų�ƥ��!",
					xmlfilename);
		nCode=-1;
		goto ReadFrmFile_EXIT;
	}
	nCode = 0;
ReadFrmFile_EXIT:
	/*
	#if defined(_AIX_) || defined(_WIN32_)
		dup2(oldstdin, 0);
	#endif
	*/
	return nCode;
	
}
void Xml::Clear()
{
	list<XmlNode*> lst;
	this->DepthScan(lst);
	list<XmlNode*>::iterator it;
	#ifdef _DEBUG
			//printf("����[%s][%d]\n", __FILE__, __LINE__);
			//printf("lst.size()=[%d]\n", lst.size());
	#endif
	for (it = lst.begin(); it != lst.end(); ++it)
	{
		XmlNode * ptr = *it;
		#ifdef _DEBUG
		//printf("ɾ���ڵ�%s\n", ptr->GetNodeName().c_str());
		#endif
		delete ptr;
	}
	m_RootPtr = NULL;
}
/*
*����·��
*	����path="/abc/efg:1", ִ�к�lst1��������AnsiStringԪ�� "abc"��"efg"
*	lst2��������int����Ԫ��0��1
*/
int Xml::SplitPath(const AnsiString & path, list<AnsiString> & lst1, list<int> &lst2)
{
	lst1.clear();
	lst2.clear();
	if (path[0] != '/')
	{
		return -1;
	}

	AnsiString ppp = path;
	ppp.concat("/");

	int start = 1;
	int len = ppp.length();
	for (int i = 1; i < len; ++i)
	{
		if (ppp[i] == '/')
		{
			AnsiString sub = ppp.substring(start, i-start);
			start = i+1;

			sub.trim();
			if (sub.length() == 0) /*���ڵ�����'/'������һ��*/
			{
				continue;
			}

			/*
			* ���� abc:1 �ֳ�abc��1
			*      abc   �ֳ�abc��0
			*/
			int sep_index = sub.GetIndexOf(':');
			if (sep_index == 0)
			{
				return -1;
			}
			if (sep_index != -1)
			{
				lst1.push_back(sub.substring(0, sep_index));
				lst2.push_back(atoi(sub.substring(sep_index+1).c_str()));
			}
			else
			{
				lst1.push_back(sub);
				lst2.push_back(0);
			}
		}
	}
	return 0;
}
/*
*���ַ���xxx����ת�崦��:
*	&ltתΪ<,&gtתΪ>,&ampתΪ&,&aposתΪ' &quotתΪ"
*/
void Xml::TransferMeaning(AnsiString & xxx)
{
	char * p = NULL;
	AnsiString retstr;
	AnsiString src, dst;

	int len = xxx.length();
	int i;
	for (i = 0; i < len; ++i)
	{
		if (xxx[i] == '&')
		{
			/*	&ltתΪ<,&gtתΪ>,&ampתΪ&,&aposתΪ' &quotתΪ" */
			if (xxx.substring(i, 3) == AnsiString("&lt"))
			{
				retstr.concat("<");
				i += 2;
			}
			else if (xxx.substring(i,3)==AnsiString("&gt"))
			{
				retstr.concat(">");
				i += 2;
			}
			else if (xxx.substring(i,4)==AnsiString("&amp"))
			{
				retstr.concat("&");
				i += 3;
			}
			else if (xxx.substring(i,5)==AnsiString("&apos"))
			{
				retstr.concat("'");
				i += 4;
			}
			else if (xxx.substring(i,5)==AnsiString("&quot"))
			{
				retstr.concat("\"");
				i += 4;
			}
			else
			{
				retstr.concat("&");
			}
		}
		else
		{
			char tmp[10];
			memset(tmp, 0, sizeof(tmp));
			tmp[0] = xxx[i];
			retstr.concat(tmp);
		}
	}
	xxx = retstr;
}
/*
*���ַ���xxx����ת�崦��:
*	<תΪ&lt,>תΪ&gt,&תΪ&amp,'תΪ&apos "תΪ&quot
*/
void Xml::TransferMeaning2(AnsiString & xxx)
{
	char * p = NULL;
	AnsiString retstr = "";
	int len = xxx.length();
	int i;
	for (i = 0; i < len; ++i)
	{
		if (xxx[i] == '<')
		{
			retstr.concat("&lt");
		}
		else if (xxx[i]=='>')
		{
			retstr.concat("&gt");	
		}
		else if (xxx[i]=='\'')
		{
			retstr.concat("&apos");	
		}
		else if (xxx[i]=='\"')
		{
			retstr.concat("&quot");	
		}
		else if (xxx[i]=='&')
		{
			retstr.concat("&amp");	
		}
		else
		{
			char tmp[10];
			::memset(tmp, 0, sizeof(tmp));
			tmp[0] = xxx[i];
			retstr.concat(tmp);
		}
	}
	xxx = retstr;
}
int Xml::GetNodeInfo(const char * path, char* NodeValue, unsigned int NodeValueSize,
									   char* NodeProperty, unsigned int NodePropertySize)
{
	if (NULL == NodeValue || NULL == NodeProperty)
	{
		return -1;
	}
	if (NULL == m_RootPtr)
	{
		return -1;
	}
	list<AnsiString> list1;
	list<int> list2;
	if (SplitPath(path, list1, list2) != 0)
	{
		return -1;
	}

	XmlNode * curptr = m_RootPtr;

	list<AnsiString>::const_iterator it1 = list1.begin();
	list<int>::const_iterator it2 = list2.begin();
	AnsiString nodename = *it1;
	int nodeindex = *it2;

	if (m_RootPtr->GetNodeName() == nodename && nodeindex == 0)
	{
		/*���ڵ��ܹ�ƥ��*/
	}
	else
	{
		return -1;
	}

	for (++it1, ++it2; it1 != list1.end(); ++it1, ++it2)
	{
		nodename = *it1;
		nodeindex = *it2;

		curptr = curptr->GetChildByNodeName(nodename.c_str(), nodeindex);
		if (curptr == NULL)
		{
			return -1;
		}
	}
	AnsiString value = curptr->GetNodeValue();
	AnsiString property = curptr->GetNodeProperty();
	/*ת��*/
	TransferMeaning(value);
	//TransferMeaning(property);

	::memset(NodeValue, 0, NodeValueSize);
	::memset(NodeProperty, 0, NodePropertySize);
	strncpy(NodeValue, value.c_str(), NodeValueSize);
	strncpy(NodeProperty, property.c_str(), NodePropertySize);
	return 0;
}
int Xml::InitRoot(const char* RootName, const char* RootValue, const char* RootProperty)
{
	if (RootName == NULL || RootValue == NULL || RootProperty == NULL)
	{
		return -1;
	}
	Clear();
	m_RootPtr = new XmlNode();
	if ( NULL == m_RootPtr)
	{
		return -1;
	}

	AnsiString val, prop;
	val = RootValue;
	prop = RootProperty;

	TransferMeaning2(val);
	//TransferMeaning2(prop);

	m_RootPtr->SetNodeName(RootName);
	m_RootPtr->SetNodeValue(val);
	m_RootPtr->SetNodeProperty(prop);
	return 0;
}
int Xml::SetNodeInfo(const char * path, const char* NodeValue, const char* NodeProperty)
{
	if (NULL == NodeValue || NULL == NodeProperty)
	{
		return -1;
	}
	if (NULL == m_RootPtr) /*�������Ѿ���ʼ��*/
	{
		return -1;
	}
	list<AnsiString> list1;
	list<int> list2;
	if (SplitPath(path, list1, list2) != 0)
	{
		return -1;
	}


	list<AnsiString>::const_iterator it1 = list1.begin();
	list<int>::const_iterator it2 = list2.begin();
	AnsiString nodename = *it1;
	int nodeindex = *it2;

	if (m_RootPtr->GetNodeName() == nodename && nodeindex == 0)
	{
		/*���ڵ��ܹ�ƥ��*/
	}
	else
	{
		return -1;
	}

	XmlNode * curptr = m_RootPtr;
	/*һ���İ�·����������*/
	for (++it1, ++it2; it1 != list1.end(); ++it1, ++it2)
	{
		nodename = *it1;
		nodeindex = *it2;

		if (curptr->AssureChildExist(nodename.c_str(), nodeindex) < 0)
		{
			return -1;
		}
		curptr = curptr->GetChildByNodeName(nodename.c_str(), nodeindex);
		if (curptr == NULL)
		{
			return -1;
		}
	}
	AnsiString val, prop;
	val =  NodeValue;
	prop = NodeProperty;

	TransferMeaning2(val);
	//TransferMeaning2(prop);

	curptr->SetNodeValue(val);
	curptr->SetNodeProperty(prop);
	return 0;
}
const char * Xml::GetRootName() const
{
	if (m_RootPtr == NULL)
	{
		return NULL;
	}
	return m_RootPtr->GetNodeName().c_str();
}
int Xml::ExportNode(const char* path, char* buffer, int buffersize, bool bNodeSelf )
{
	if (buffer == NULL)
	{
		return -1;
	}
	::memset(buffer, 0, buffersize);

	if (NULL == m_RootPtr)
	{
		return -1;
	}
	list<AnsiString> list1;
	list<int> list2;
	if (SplitPath(path, list1, list2) != 0)
	{
		return -1;
	}

	XmlNode * curptr = m_RootPtr;

	list<AnsiString>::const_iterator it1 = list1.begin();
	list<int>::const_iterator it2 = list2.begin();
	AnsiString nodename = *it1;
	int nodeindex = *it2;

	if (m_RootPtr->GetNodeName() == nodename && nodeindex == 0)
	{
		/*���ڵ��ܹ�ƥ��*/
	}
	else
	{
		return -1;
	}

	for (++it1, ++it2; it1 != list1.end(); ++it1, ++it2)
	{
		nodename = *it1;
		nodeindex = *it2;

		curptr = curptr->GetChildByNodeName(nodename.c_str(), nodeindex);
		if (curptr == NULL)
		{
			return -1;
		}
	}
	//������, curptrָ����pathָ���Ľڵ�
	XmlNode * startnode = curptr;

	stack<DEPTH_SCAN_STACK_ELE_TYPE> stk;
	DEPTH_SCAN_STACK_ELE_TYPE stk_ele, stk_ele2;
	list<XmlNode*>::const_iterator it;
	curptr = NULL;

	stk_ele.NodePtr = startnode;
	stk_ele.It = startnode->GetNodeChildList().begin();
	stk.push(stk_ele);

	AnsiString xmlstring =  "";
	if (bNodeSelf)
	{
		xmlstring.concat("<");
		xmlstring.concat(stk_ele.NodePtr->GetNodeName());
		if (stk_ele.NodePtr->GetNodeProperty().length() == 0)
		{
			xmlstring.concat(">");
		}
		else
		{
			xmlstring.concat(" ");
			xmlstring.concat(stk_ele.NodePtr->GetNodeProperty());
			xmlstring.concat(">");
		}
	}

	while (!stk.empty())
	{
		stk_ele2 = stk.top();
		if (stk_ele2.NodePtr->GetNodeChildList().empty())	/*Ҷ�ӽڵ�*/
		{
			if (stk_ele2.NodePtr->GetNodeValue().length() > 0)
			{
				xmlstring.concat(">");
				/*дҶ�ӽڵ��ֵ*/
				xmlstring.concat(stk_ele2.NodePtr->GetNodeValue());
				/*дҶ�ӽڵ�Ľ�������*/
				xmlstring.concat("</");
				xmlstring.concat(stk_ele2.NodePtr->GetNodeName());
				xmlstring.concat(">\n");
			}
			else
			{
				xmlstring.concat("/>\n");
			}

			stk.pop();
		}
		else	/*����Ҷ��*/
		{
			it = stk_ele2.It;
			if (it == stk_ele2.NodePtr->GetNodeChildList().end()) /*���к��Ӷ�������*/
			{
				xmlstring.concat("</");
				xmlstring.concat(stk_ele2.NodePtr->GetNodeName());
				xmlstring.concat(">");
				stk.pop();
			}
			else /*������һ������*/
			{
				//ȡ���ӽڵ�ָ�롢��ʼ���ӽڵ�ĵ�����
				stk_ele.NodePtr = *it;
				stk_ele.It = stk_ele.NodePtr->GetNodeChildList().begin();
				//�޸�ջ���ڵ��It��ֵ
				stk.pop();
				++(stk_ele2.It);
				stk.push(stk_ele2);
				//�����С��ѹջ
				if (stk_ele.NodePtr->GetNodeChildList().empty())//��С����Ҷ�ӽڵ�
				{
					xmlstring.concat("<");
					xmlstring.concat(stk_ele.NodePtr->GetNodeName());
					if (stk_ele.NodePtr->GetNodeProperty().length() == 0)
					{
					}
					else
					{
						xmlstring.concat(" ");
						xmlstring.concat(stk_ele.NodePtr->GetNodeProperty());
					}
				}
				else
				{
					xmlstring.concat("<");
					xmlstring.concat(stk_ele.NodePtr->GetNodeName());
					if (stk_ele.NodePtr->GetNodeProperty().length() == 0)
					{
						xmlstring.concat(">");
					}
					else
					{
						xmlstring.concat(" ");
						xmlstring.concat(stk_ele.NodePtr->GetNodeProperty());
						xmlstring.concat(">");
					}
				}
				stk.push(stk_ele);
			}
		}
	}
	strncpy(buffer, xmlstring.c_str(), buffersize);

	return 0;
}