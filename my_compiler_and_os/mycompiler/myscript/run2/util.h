#ifndef _UTIL_H_INCLUDED_
#define _UTIL_H_INCLUDED_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <list>
using namespace std;
#include <regex.h>
#include <deque>
#include "var.h"

namespace lnb {

/*
 * ������ʽƥ��
 * str: ��Ҫ�����ַ���
 * pattern: ��Ҫƥ�������ʽ
 * icase:   ��ʾ��ƥ��������Ƿ����ִ�Сд
 * ���ִ�гɹ�
 *  startpos�ﱣ�����ÿ��ƥ���Ӵ��Ŀ�ʼ�±�
 *  endpos�ﱣ�����ÿ��ƥ���Ӵ��Ľ����±�, ���������Ӵ���, Ҳ����˵ endpos[i]-startpos[i]ֱ�ӵõ��Ӵ��ĳ���
 */
bool re(const string & str, const string & pattern, bool icase, list<int> & startpos, list<int> & endpos);
void StringUnescape(const string & src, string & dest);
void StringEscape(const string & src, string & dest);

};

#endif
