#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "AnsiString.h"

AnsiString::AnsiString():m_s("")
{
}
AnsiString::AnsiString(const char * pstr):m_s("")
{
	if (NULL != pstr)
	{
		m_s = pstr;
	}
}
AnsiString::AnsiString(const AnsiString& s)
{
	m_s = s.m_s;
}
AnsiString::~AnsiString()
{
}
void AnsiString::concat(const char *pstr)
{
	if (NULL == pstr)
	{
		return;
	}
	m_s = m_s + pstr;
}
void AnsiString::concat(const AnsiString & s)
{
	m_s = m_s + s.m_s;
}
unsigned int AnsiString::length() const
{
	return m_s.length();
}
AnsiString & AnsiString::operator=(const AnsiString& s)
{
	m_s = s.m_s;
	return *this;
}
bool AnsiString::operator==(const AnsiString& s) const
{
	if (m_s == s.m_s)
	{
		return TRUE;
	}
	return FALSE;
}
char AnsiString::operator[](int index) const
{
	if (index < 0 || index >= m_s.length())
	{
		return 0;
	}
	return m_s[index];
}
const char * AnsiString::c_str() const
{
	return  m_s.c_str();
}
bool operator==(const char * pstr, const AnsiString & s)
{
	if (pstr == NULL)
	{
		return FALSE;
	}
	if (s==pstr)
	{
		return TRUE;
	}
	return FALSE;
}
  /*缩小缓冲区，避免不必要的内存使用*/
void AnsiString::trimToSize()
{
}
void AnsiString::clear()
{
	m_s.clear();
}
AnsiString AnsiString::substring(unsigned int start, unsigned int len) const
{
	AnsiString xxx;
	xxx.m_s = this->m_s.substr(start, len);
	return xxx;
}
AnsiString AnsiString::substring(unsigned int start) const
{
	AnsiString xxx;
	xxx.m_s = m_s.substr(start);
	return xxx;
}
void AnsiString::ltrim()
{
	int offset = 0;
	int len = m_s.length();
	for (; offset < len;++offset)
	{
		if (m_s[offset] == ' ' || m_s[offset] == '\t')
		{
		}
		else
		{
			break;
		}
	}
	m_s = m_s.substr(offset);
}
void AnsiString::rtrim()
{
	int len = m_s.length();
	for (; len>0; --len)
	{
		if (m_s[len-1] == ' ' || m_s[len-1] == '\t')
		{
		}
		else
		{
			break;
		}
	}
	m_s = m_s.substr(0, len);
}
void AnsiString::trim()
{
	ltrim();
	rtrim();
}
int AnsiString::GetIndexOf(char c) const
{
	int len = m_s.length();
	int i;
	for (i = 0; i < len; ++i)
	{
		if (m_s[i] == c)
		{
			return i;
		}
	}
	return -1;
}
