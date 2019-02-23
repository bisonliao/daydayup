#include <stdlib.h>
#include <string.h>
#include <stdio.h> 
#include "AnsiString.h"

AnsiString::AnsiString()
{
	m_bufsize = 1;
	m_buffer = new char[m_bufsize];	
	if (m_buffer == NULL)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
			__FILE__,
			__LINE__);
		exit(-1);
	}
	memset(m_buffer, 0, m_bufsize);
}
AnsiString::AnsiString(const char * pstr)
{
	if (NULL == pstr)
	{
		m_bufsize = 1;
		m_buffer = new char[1];	
		if (m_buffer == NULL)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
				__FILE__,
				__LINE__);
			exit(-1);
		}
	}
	else
	{
		m_bufsize = strlen(pstr) + 5;
		m_buffer = new char[m_bufsize];
		if (m_buffer == NULL)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
				__FILE__,
				__LINE__);
			exit(-1);
		}
		memset(m_buffer, 0, m_bufsize);
		strcpy(m_buffer, pstr);
	}
}
AnsiString::AnsiString(const AnsiString& s)
{
	m_bufsize = s.length() + 5;
	m_buffer = new char[m_bufsize];
	if (m_buffer == NULL)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
			__FILE__,
			__LINE__);
		exit(-1);
	}
	memset(m_buffer, 0, m_bufsize);
	strcpy(m_buffer, s.m_buffer);
}
AnsiString::~AnsiString()
{
	if (m_buffer != NULL)
	{
		delete[] m_buffer;
		m_buffer = NULL;
		m_bufsize = 0;
	}
}
void AnsiString::concat(const char *pstr)
{
	if (NULL == pstr)
	{
		return;
	}
	if (strlen(m_buffer) + strlen(pstr) >= m_bufsize)
	{
		m_bufsize = strlen(m_buffer) + strlen(pstr) + 5;
		m_buffer = (char*)realloc(m_buffer, m_bufsize);
		if (m_buffer == NULL)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
				__FILE__,
				__LINE__);
			exit(-1);
		}
	}
	strcat(m_buffer, pstr);
}
void AnsiString::concat(const AnsiString & s)
{
	if (strlen(m_buffer) + s.length() >=  m_bufsize)
	{
		m_bufsize = strlen(m_buffer) + s.length() + 5;
		m_buffer = (char*)realloc(m_buffer, m_bufsize);
		if (m_buffer == NULL)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
				__FILE__,
				__LINE__);
			exit(-1);
		}
	}
	strcat(m_buffer, s.c_str());
}
unsigned int AnsiString::length() const
{
	return strlen(m_buffer);
}
AnsiString & AnsiString::operator=(const AnsiString& s)
{
	/*已有的空间不够或者太大,都会重新分配内存*/
	int len = s.length() + 5;
	if (len > m_bufsize	|| 
		(m_bufsize > len ) && (m_bufsize - len ) > (len / 3))
	{
		m_bufsize = len;
		m_buffer = (char*)realloc(m_buffer, m_bufsize);
		if (m_buffer == NULL)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
				__FILE__,
				__LINE__);
			exit(-1);
		}
	}
	memset(m_buffer, 0, m_bufsize);
	strcpy(m_buffer, s.m_buffer);
	return *this;
}
bool AnsiString::operator==(const AnsiString& s) const
{
	if (strcmp(s.m_buffer, m_buffer) == 0)
	{
		return TRUE;
	}
	return FALSE;
}
char AnsiString::operator[](int index) const
{
	if (index < 0 || index >= strlen(m_buffer))
	{
		return 0;
	}
	return m_buffer[index];
}
const char * AnsiString::c_str() const
{
	return  m_buffer;
}
bool operator==(const char * pstr, const AnsiString & s)
{
	if (pstr == NULL)
	{
		return FALSE;
	}
	if (strcmp(pstr, s.c_str()) == 0)
	{
		return TRUE;
	}
	return FALSE;
}
  /*缩小缓冲区，避免不必要的内存使用*/
void AnsiString::trimToSize()
{
	int len = strlen(m_buffer);
	if (m_bufsize - len > 30)
	{
		m_bufsize = len + 5;
		m_buffer = (char*)realloc(m_buffer, m_bufsize);
	}
}
void AnsiString::clear()
{
	delete[] m_buffer;
	m_buffer = NULL;

	m_bufsize = 1;
	m_buffer = new char[m_bufsize];	
	if (m_buffer == NULL)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
			__FILE__,
			__LINE__);
		exit(-1);
	}
	memset(m_buffer, 0, m_bufsize);
}
AnsiString AnsiString::substring(unsigned int start, unsigned int len) const
{
	if (start >= this->length())
	{
		return (AnsiString)"";
	}
	char * tmp = new char[this->length() + 1];
	if (NULL == tmp)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
			__FILE__,
			__LINE__);
		exit(-1);
	}
	strcpy(tmp, (this->c_str() + start));
	if ( (start + len) < this->length())
	{
		tmp[len] = 0;
	}
	AnsiString s = (AnsiString)tmp;
	delete[] tmp;
	return s;
}
AnsiString AnsiString::substring(unsigned int start) const
{
	if (start >= this->length())
	{
		return (AnsiString)"";
	}
	unsigned int len = this->length() - start;
	char * tmp = new char[this->length() + 1];
	if (NULL == tmp)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
			__FILE__,
			__LINE__);
		exit(-1);
	}
	strcpy(tmp, (this->c_str() + start));
	if ( (start + len) < this->length())
	{
		tmp[len] = 0;
	}
	AnsiString s = (AnsiString)tmp;
	delete[] tmp;
	return s;
}
void AnsiString::ltrim()
{
    int len;
    int start;
    int i;

    if (NULL == m_buffer) {
        return;
    }
    len = strlen(m_buffer);

    /*找到第一个不是空格的字符的位置*/
    start = 0;
    while ( (m_buffer[start] == ' ' || m_buffer[start] == '\t') && start < len) {
        start++;
    }
    /*逐个拷贝*/
    if (start > 0) {
        i = 0;
        while (start < len) {
            m_buffer[i] = m_buffer[start];
            i++;
            start++;
        }
        m_buffer[i] = 0;
    }
}
void AnsiString::rtrim()
{
    int len;

    if (NULL == m_buffer) {
        return;
    }

    len = strlen(m_buffer);
    len--;
    while ((m_buffer[len] == ' ' || m_buffer[len] == '\t') &&
        len >= 0) {
        len--;
    }
    m_buffer[len + 1] = 0;
}
void AnsiString::trim()
{
	ltrim();
	rtrim();
}
int AnsiString::GetIndexOf(char c) const
{
	int len = strlen(m_buffer);
	for (int i = 0; i < len; ++i)
	{
		if (m_buffer[i] == c)
		{
			return i;
		}
	}
	return -1;
}
