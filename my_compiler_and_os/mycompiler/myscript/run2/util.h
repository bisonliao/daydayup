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
 * 正则表达式匹配
 * str: 需要检查的字符串
 * pattern: 需要匹配的正则式
 * icase:   表示在匹配过程中是否区分大小写
 * 如果执行成功
 *  startpos里保存的是每个匹配子串的开始下标
 *  endpos里保存的是每个匹配子串的结束下标, 不包含在子串中, 也就是说 endpos[i]-startpos[i]直接得到子串的长度
 */
bool re(const string & str, const string & pattern, bool icase, list<int> & startpos, list<int> & endpos);
void StringUnescape(const string & src, string & dest);
void StringEscape(const string & src, string & dest);

};

#endif
