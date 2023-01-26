/*
��һ���ַ��� s ������ÿһ����ĸ�Ĵ�д��Сд��ʽ ͬʱ ������ s �У��ͳ�����ַ��� s �� ���� �ַ������ȷ�˵��"abABB" �������ַ�������Ϊ 'A' �� 'a' ͬʱ�����ˣ��� 'B' �� 'b' Ҳͬʱ�����ˡ�Ȼ����"abA" ���������ַ�����Ϊ 'b' �����ˣ��� 'B' û�г��֡�
����һ���ַ��� s �����㷵�� s ��� �������ַ��� ������ж���𰸣����㷵�� ���� ���ֵ�һ��������������������ַ��������㷵��һ�����ַ�����

ʾ�� 1��
���룺s = "YazaAay"
�����"aAa"
���ͣ�"aAa" ��һ�������ַ�������Ϊ����Ӵ��н���һ����ĸ����Сд��ʽ 'a' �ʹ�д��ʽ 'A' Ҳͬʱ�����ˡ�
"aAa" ������������ַ�����

ʾ�� 2��
���룺s = "Bb"
�����"Bb"
���ͣ�"Bb" �������ַ�������Ϊ 'B' �� 'b' �������ˡ������ַ���Ҳ��ԭ�ַ��������ַ�����

ʾ�� 3��
���룺s = "c"
�����""
���ͣ�û���������ַ�����

ʾ�� 4��
���룺s = "dDzeE"
�����"dD"
���ͣ�"dD" �� "eE" ������������ַ�����
�����ж���������ַ��������� "dD" ����Ϊ�����ֵ����硣
 
��ʾ��
1 <= s.length <= 100
s ֻ������д��СдӢ����ĸ��
===============================================================
�ҵ�˼·��
�����ַ����е�ÿ��Ԫ�أ������Ԫ���ڸ��ַ����д��ڶ�Ӧ�Ĵ�/СдԪ�أ��ǰѸ�Ԫ�غ����ж�Ӧ��Ԫ�أ������ж������λ������ǣ�
һ�����������ÿ��λ�ö�������ˣ��ǵ�ǰ�ַ������������ַ�����
����е�λ��û�б���ǣ��ͻᷢ��û�б�ǵ�λ�ö�Ӧ��Ԫ���ǡ��������� ������Ԫ��ʹ�ø��ַ��������������ַ�����
�������ַ���ֻ���ܳ����������ġ�������Ԫ�����ҡ� ����������Ԫ�ذѵ�ǰ�ַ����ָ��������ַ�������Щ���ַ����ǿ��ܵ��������ַ���
�����ݹ顣

���������ô�cд�ģ�������Ҫ�ڴ涯̬��������������c++�ɣ��ƽ���
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
    // index��ʣ�µľ�������Ԫ�ص��±���
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
