#include "util.h"
/*
 * ������ʽƥ��
 * str: ��Ҫ�����ַ���
 * pattern: ��Ҫƥ�������ʽ
 * icase:   ��ʾ��ƥ��������Ƿ����ִ�Сд
 * ���ִ�гɹ�
 *  startpos�ﱣ�����ÿ��ƥ���Ӵ��Ŀ�ʼ�±�
 *  endpos�ﱣ�����ÿ��ƥ���Ӵ��Ľ����±�, ���������Ӵ���, Ҳ����˵ endpos[i]-startpos[i]ֱ�ӵõ��Ӵ��ĳ���
 */
bool lnb::re(const string & str, const string & pattern, bool icase, 
			list<int> & startpos, list<int> & endpos)
{
	startpos.clear();
	endpos.clear();

	regex_t reg;
	int cflag = REG_EXTENDED;
	if (icase)
	{
		cflag = cflag | REG_ICASE;
	}
	if (0 != regcomp(&reg, pattern.c_str(), cflag))
	{
		return false;	
	}


	regmatch_t match[1];
	memset(&match, 0, sizeof(match));

	int offset = 0;

	while (1)
	{
    	int result = regexec(&reg,  str.c_str()+offset, 1, match, 0);
		if (REG_NOMATCH == result)
		{
       		regfree(&reg);
	   		return true;
		}
		else if (0 != result)
		{
       		regfree(&reg);
	   		return false;
		}

		if (match[0].rm_so == -1)
		{
			break;
		}
		startpos.push_back(match[0].rm_so + offset);
		endpos.push_back(match[0].rm_eo + offset);

		offset += match[0].rm_eo;

		if (offset >= str.length())
		{
			break;
		}
	}
    regfree(&reg);
	return true;
}
void lnb::StringUnescape(const string & src, string & dest)
{
	dest = "";
	dest.reserve(src.length());
	int i;
	int iLen = src.length() ;
	for (i = 0; i < iLen; ++i)
	{
		if (src[i] == '\\')
		{
			if ( (i+1) >= iLen) //���һ���ַ�
			{
				//����ȷ������,ֱ�Ӷ���
				break;
			}

			//����һ���ַ������
			char cc = src[i+1];
			if ('\\' == cc)
			{
				dest.append(1, '\\'); ++i;
			}
			else if ('n' == cc)
			{
				dest.append(1, '\n'); ++i;
			}
			else if ('r' == cc)
			{
				dest.append(1, '\r'); ++i;
			}
			else if ('t' == cc)
			{
				dest.append(1, '\t'); ++i;
			}
			else if ('b' == cc)
			{
				dest.append(1, '\b'); ++i;
			}
			else if ('"' == cc)
			{
				dest.append(1, '"'); ++i;
			}
			else
			{
				//����ȷ������,ֱ�Ӷ���
			}
		}
		else
		{
			dest.append(1, src[i]);
		}
	}
}
void lnb::StringEscape(const string & src, string & dest)
{
	dest = "";
	dest.reserve(src.length()+100);

	int i;
	int iLen = src.length() ;
	for (i = 0; i < iLen; ++i)
	{
		char cc = src[i];	
		if ('\\' == cc)
		{
			dest.append("\\\\");
		}
		else if ('\n' == cc)
		{
			dest.append("\\n");
		}
		else if ('\r' == cc)
		{
			dest.append("\\r");
		}
		else if ('\t' == cc)
		{
			dest.append("\\t");
		}
		else if ('\b' == cc)
		{
			dest.append("\\b");
		}
		else if ('"' == cc)
		{
			dest.append("\\\"");
		}
		else
		{
			dest.append(1, src[i]);
		}
	}
}
