#ifndef _TOKEN_H_INCLUDED_
#define _TOKEN_H_INCLUDED_

#include <string>
using namespace std;

#define IF 				256
#define THEN 			257
#define	ELSE 			258
#define ENDIF			259
#define WHILE			260
#define DO				261
#define ENDWHILE		262
#define BEGIN_SCRIPT	263
#define END_SCRIPT		264
#define RETURN			265
#define CONTINUE		266
#define BREAK			267

#define VAR				268
#define CONST_STRING	269
#define CONST_FLOAT		270

#define GT				271
#define GE				272
#define LE				273
#define LT				274
#define EQ				275
#define ADD				276
#define SUB				277
#define MUL				278
#define DIV				279
#define LBRK			280
#define RBRK			281
#define NOT				282

#define ID				283
#define COMMA			284
#define CONST_INT		285
#define NE				286

#define FUNCTION			300

#define LOFFSET			301 /*左方括号，用于数组下标开始*/
#define ROFFSET			302 /*右方括号，用于数组下标结束*/

class YYLVAL
{
public:
	std::string id_val;
	double float_val;
	long int_val;
	std::string string_val;
	unsigned int lineno;

	const YYLVAL & operator=(const YYLVAL& another);
	YYLVAL(const YYLVAL& another);
	YYLVAL();
};

class CToken
{
public:
	CToken(int token, const YYLVAL &yylval);
	CToken(int token);
	CToken();
	CToken(const CToken & another);
	const CToken & operator=(const CToken&another);

	int GetToken() const; //返回单词的ID
	void GetYYLVAL(YYLVAL & yylval) const; //返回单词的属性

	int GetPriorityIN() const; //返回操作符单词的在栈内的优先级，如果是非操作符，返回－1
	int GetPriorityOUT() const; //返回操作符单词的在栈外的优先级，如果是非操作符，返回－1
	bool IsExprOperator() const; //是否是表达式操作符
	bool IsExprOperant() const; //是否式表达式操作数
	int  GetOprntNumNeed() const; //如果是操作符，返回需要的操作数个数，否则返回-1
	string ToString() const;



	YYLVAL m_yylval;
	int	m_token;
};

#endif
