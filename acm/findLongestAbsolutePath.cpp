/**
 * https://leetcode.cn/problems/longest-absolute-file-path/
 * 从某种约定的输入格式里，找出绝对路径最长的文件
 * 解题思路：
 * 用一个栈，保存一路过来的父节点，每遇到一个文件，就拼凑出绝对路径，并保存到结果里
*/
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <stack>
#include <map>

using namespace std;

struct Node_t
{
    string expr;
    int level;
};
struct Token_t
{
    enum
    {
        TOKEN_FILE = 1,
        TOKEN_DIR = 2,
        TOKEN_TAB = 3,
        TOKEN_NEWLINE = 4,
    }  token;
    string expr;
};

int pushFileOrDir(string &expr, vector<Token_t> & tokens)
{
    if (expr.length() > 0)
    {
        Token_t t;
        if (expr.find(".") != string::npos)
        {
            t.token = Token_t::TOKEN_FILE;
        }
        else
        {
            t.token = Token_t::TOKEN_DIR;
        }
        
        t.expr = expr;
        tokens.push_back(t);
        expr = "";
    }
    return 0;
}


int input2token(const char * input, vector<Token_t> & tokens)
{
    string expr;
    int i;
    if (input == NULL) { return -1;}
    tokens.clear();
    for (i = 0; i < strlen(input); ++i)
    {
        char c = input[i];
        if (c == '\t')
        {
            pushFileOrDir(expr, tokens);
            Token_t t;
            t.token = Token_t::TOKEN_TAB;
            t.expr = "\\t";
            tokens.push_back(t);          
        }
        else if (c == '\n')
        {
            pushFileOrDir(expr, tokens);
            Token_t t;
            t.token = Token_t::TOKEN_NEWLINE;
            t.expr = "\\n";
            tokens.push_back(t);
        }
        else
        {
            expr = expr + c;
        }
    }
    if (expr.length() > 0)
    {
        pushFileOrDir(expr, tokens);
    }
    #if 0
    vector<Token_t>::const_iterator it;
    for (it = tokens.begin(); it != tokens.end(); ++it)
    {
        printf("%d,%s\n", it->token, it->expr.c_str());
    }
    #endif 
    return 0;
}

int findLongestAbsolutePath(const char * input, vector<string> & outputPath)
{
    vector<Token_t> tokens;
    if (input2token(input, tokens)) { return -1;}
    vector<Token_t>::const_iterator it;
    vector<Node_t> parentNodes; // actual it is a stack
    
    outputPath.clear();
    int level = 0;
    for (it = tokens.begin(); it != tokens.end(); ++it)
    {
        const Token_t & token = *it;
        if (token.token == Token_t::TOKEN_TAB)
        {
            level++;
        }
        else if (token.token == Token_t::TOKEN_NEWLINE)
        {
            level = 0;
        }
        else if (token.token == Token_t::TOKEN_DIR || token.token == Token_t::TOKEN_FILE)
        {
            if (level > 0)
            {
                while (parentNodes.size() > 0 && parentNodes.back().level >= level)
                {
                    parentNodes.pop_back();
                }
                if (parentNodes.size() < 1)
                {
                    fprintf(stderr, "invalid input:%s\n", token.expr.c_str());
                    return -1;
                }
            }
            if (token.token == Token_t::TOKEN_DIR)
            {
                Node_t node;
                node.level = level;
                node.expr = token.expr;
                parentNodes.push_back(node);
            }
            else
            {
                string path = "/";
                vector<Node_t>::const_iterator it2;
                for (it2 = parentNodes.begin(); it2 != parentNodes.end(); ++it2)
                {
                    path.append(it2->expr);
                    path.append("/");
                }
                path.append(token.expr);
                printf("%s\n", path.c_str());
                if (outputPath.size() == 0)
                {
                    outputPath.push_back(path);
                }
                else
                {
                    if (outputPath.back().length() < path.length())
                    {
                        outputPath.clear();
                        outputPath.push_back(path);
                    }
                    else if (outputPath.back().length() == path.length())
                    {
                        outputPath.push_back(path);
                    }
                }

            }
        }

    }
    return 0;
}
int main()
{
    char input[1024];
    vector<string> result;
    vector<string>::const_iterator it;

    ////////////////////////////////////////
    strcpy(input, "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext");
    if (findLongestAbsolutePath(input, result))
    {
        return -1;
    }
    
    for (it = result.begin(); it != result.end(); ++it)
    {
        printf("[%d, %s]\n", it->length(), it->c_str());
    }
    ////////////////////////////////////////////////
    strcpy(input, "file1.txt\nfile2.txt\nlongfile.txt");
    if (findLongestAbsolutePath(input, result))
    {
        return -1;
    }

    for (it = result.begin(); it != result.end(); ++it)
    {
        printf("[%d, %s]\n", it->length(), it->c_str());
    }
    return 0;

}
