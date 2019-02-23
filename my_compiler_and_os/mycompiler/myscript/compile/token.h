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

#define LOFFSET			301 /*�����ţ����������±꿪ʼ*/
#define ROFFSET			302 /*�ҷ����ţ����������±����*/

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

	int GetToken() const; //���ص��ʵ�ID
	void GetYYLVAL(YYLVAL & yylval) const; //���ص��ʵ�����

	int GetPriorityIN() const; //���ز��������ʵ���ջ�ڵ����ȼ�������Ƿǲ����������أ�1
	int GetPriorityOUT() const; //���ز��������ʵ���ջ������ȼ�������Ƿǲ����������أ�1
	bool IsExprOperator() const; //�Ƿ��Ǳ��ʽ������
	bool IsExprOperant() const; //�Ƿ�ʽ���ʽ������
	int  GetOprntNumNeed() const; //����ǲ�������������Ҫ�Ĳ��������������򷵻�-1
	string ToString() const;



	YYLVAL m_yylval;
	int	m_token;
};

#endif
