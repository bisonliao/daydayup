/*
当一个字符串 s 包含的每一种字母的大写和小写形式 同时 出现在 s 中，就称这个字符串 s 是 美好 字符串。比方说，"abABB" 是美好字符串，因为 'A' 和 'a' 同时出现了，且 'B' 和 'b' 也同时出现了。然而，"abA" 不是美好字符串因为 'b' 出现了，而 'B' 没有出现。
给你一个字符串 s ，请你返回 s 最长的 美好子字符串 。如果有多个答案，请你返回 最早 出现的一个。如果不存在美好子字符串，请你返回一个空字符串。

示例 1：
输入：s = "YazaAay"
输出："aAa"
解释："aAa" 是一个美好字符串，因为这个子串中仅含一种字母，其小写形式 'a' 和大写形式 'A' 也同时出现了。
"aAa" 是最长的美好子字符串。

示例 2：
输入：s = "Bb"
输出："Bb"
解释："Bb" 是美好字符串，因为 'B' 和 'b' 都出现了。整个字符串也是原字符串的子字符串。

示例 3：
输入：s = "c"
输出：""
解释：没有美好子字符串。

示例 4：
输入：s = "dDzeE"
输出："dD"
解释："dD" 和 "eE" 都是最长美好子字符串。
由于有多个美好子字符串，返回 "dD" ，因为它出现得最早。
 
提示：
1 <= s.length <= 100
s 只包含大写和小写英文字母。
===============================================================
我的思路：
遍历字符串中的每个元素，如果该元素在该字符串中存在对应的大/小写元素，那把该元素和所有对应的元素（可能有多个）的位置做标记，
一趟下来，如果每个位置都被标记了，那当前字符串就是完美字符串，
如果有的位置没有被标记，就会发现没有标记的位置对应的元素是“异数”， 这样的元素使得该字符串不能是完美字符串。
完美子字符串只可能出现在这样的“异数”元素左右。 即“异数”元素把当前字符串分割出多个子字符串，这些子字符串是可能的完美子字符串
继续递归。

本来想坚持用纯c写的，但是需要内存动态的容器，还是用c++吧，破戒了
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <iterator>

using namespace std;

int findPerfectSubStr(const char * s, std::vector<string> & sub)
{
    std::string a;
   
    std::set<int> index;
    std::map<char, std::vector<int>*> chr2pos;
    int length = strlen(s);

    // index the string
    int i;
    for (i = 0; i < length; ++i)
    {
        index.insert(i);
        if (chr2pos.find(s[i]) == chr2pos.end())
        {
            std::vector<int>* pos = new std::vector<int>;
            pos->push_back(i);
            chr2pos.insert(std::pair<char, std::vector<int>*>(s[i], pos));           
        }
        else
        {
            chr2pos.find(s[i])->second->push_back(i);
        }
    }
    //delete the position of element who has friend
    for (i = 0; i < length; ++i)
    {
        if (s[i]>='a' && s[i] <= 'z')
        {
            char c = s[i]-'a'+'A';
            if (chr2pos.find(c) != chr2pos.end()) // friend exist
            {
                index.erase(i);
                std::vector<int>* pos = chr2pos.find(c)->second;
                std::vector<int>::iterator it ;
                for (it = pos->begin(); it != pos->end(); ++it)
                {
                    if (index.find(*it) != index.end())
                    {
                        index.erase(*it);
                    }
                    
                }
                
            }

        }
        else if (s[i]>='A' && s[i] <= 'Z')
        {
            char c = s[i]-'A'+'a';
            if (chr2pos.find(c) != chr2pos.end()) // friend exist
            {
                index.erase(i);
                std::vector<int>* pos = chr2pos.find(c)->second;
                std::vector<int>::iterator it ;
                for (it = pos->begin(); it != pos->end(); ++it)
                {
                    if (index.find(*it) != index.end())
                    {
                        index.erase(*it);
                    }
                }
            }
        }
        else
        {
            printf("invalid char at position%d:%c\n", i, s[i]);
            return -1;
        }
    }
    std::map<char, std::vector<int>*>::iterator it2;
    for (it2 = chr2pos.begin(); it2 != chr2pos.end(); ++it2)
    {
        free(it2->second);
    }
    // index里剩下的就是奇异元素的下标了
    if (index.size() == 0)
    {
        sub.push_back(string(s));
        return 0;
    }
    string ss = s;
    int start = 0;
    std::set<int>::iterator it3;
    for (it3 = index.begin(); it3 != index.end(); it3++)
    {
        if (start < *it3)
        {
            string sss = ss.substr(start, *it3 - start); 
            findPerfectSubStr(sss.c_str(), sub);
        }
        start = *it3 + 1;
        if (start >= length)
        {
            break;
        }
    }
    if (start < length)
    {
        string sss = ss.substr(start, length - start); 
        findPerfectSubStr(sss.c_str(), sub);
    }
    
    return 0;
    
}
int main()
{
    std::vector<string> sub;
    findPerfectSubStr("YazaAay", sub);
    std::vector<string>::iterator it;
    for (it = sub.begin(); it != sub.end(); it++)
    {
        printf("[%s]\n", it->c_str());
    }
    return 0;
}
