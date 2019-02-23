#include "token.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const YYLVAL & YYLVAL::operator=(const YYLVAL& another)
{
	this->id_val = another.id_val;
	this->string_val = another.string_val;
	this->float_val = another.float_val;
	this->int_val = another.int_val;
	this->lineno = another.lineno;
	return *this;
}
YYLVAL::YYLVAL(const YYLVAL& another)
{
	this->id_val = another.id_val;
	this->string_val = another.string_val;
	this->float_val = another.float_val;
	this->int_val = another.int_val;
	this->lineno = another.lineno;
}
YYLVAL::YYLVAL()
{
	this->id_val = "";
	this->string_val = "";
	this->float_val = 0.0;
	this->int_val = 0;
	this->lineno = 0;
}
string CToken::ToString() const
{
	string rts;
	if (m_token == IF) { rts = "IF";}
	else if (m_token == THEN) { rts = "THEN";}
	else if (m_token == ELSE) { rts = "ELSE";}
	else if (m_token == ENDIF) { rts = "ENDIF";}
	else if (m_token == WHILE) { rts = "WHILE";}
	else if (m_token == DO) { rts = "DO";}
	else if (m_token == ENDWHILE) { rts = "ENDWHILE";}
	else if (m_token == BEGIN_SCRIPT) { rts = "BEGIN_SCRIPT";}
	else if (m_token == END_SCRIPT) { rts = "END_SCRIPT";}
	else if (m_token == RETURN) { rts = "RETURN";}
	else if (m_token == CONTINUE) { rts = "CONTINUE";}
	else if (m_token == BREAK) { rts = "BREAK";}
	else if (m_token == VAR) { 
	 	rts = m_yylval.id_val;
	}
	else if (m_token == ID) { 
	 	rts = m_yylval.id_val;
	}
	else if (m_token == CONST_STRING) 
	{ 
		rts = "#" + m_yylval.string_val;
	}
	else if (m_token == CONST_FLOAT) 
	{ 
		char buf[100];
		sprintf(buf, "%f", m_yylval.float_val);
		rts = string("%") + buf;
	}
	else if (m_token == CONST_INT) 
	{ 
		char buf[100];
		sprintf(buf, "%d", m_yylval.int_val);
		rts = string("^") + buf;
	}
	else if (m_token == GT) { rts = "GT";}
	else if (m_token == GE) { rts = "GE";}
	else if (m_token == LE) { rts = "LE";}
	else if (m_token == LT) { rts = "LT";}
	else if (m_token == EQ) { rts = "EQ";}
	else if (m_token == NE) { rts = "NE";}
	else if (m_token == ADD) { rts = "ADD";}
	else if (m_token == SUB) { rts = "SUB";}
	else if (m_token == MUL) { rts = "MUL";}
	else if (m_token == DIV) { rts = "DIV";}
	else if (m_token == LBRK) { rts = "LBRK";}
	else if (m_token == RBRK) { rts = "RBRK";}
	else if (m_token == NOT) { rts = "NOT";}
	else if (m_token == FUNCTION) { rts = "FUNCTION";}

	return rts;
}
int CToken::GetOprntNumNeed() const
{
	if (m_token == GT) { return 2; }
	else if (m_token == GE) { return 2; }
	else if (m_token == LE) { return 2; }
	else if (m_token == LT) { return 2; }
	else if (m_token == EQ) { return 2; }
	else if (m_token == NE) { return 2; }
	else if (m_token == ADD) { return 2; }
	else if (m_token == SUB) { return 2; }
	else if (m_token == MUL) { return 2; }
	else if (m_token == DIV) { return 2; }
	else if (m_token == NOT) { return 1; }
	else { return -1;}



}
CToken::CToken()
{
	m_token = -1; //运行时中间变量
}
CToken::CToken(const CToken & another)
{
	m_token = another.m_token;
	m_yylval = another.m_yylval;
}
const CToken & CToken::operator=(const CToken&another)
{
	m_token = another.m_token;
	m_yylval = another.m_yylval;
}

CToken::CToken(int token, const YYLVAL &yylval)
{
	m_token = token;
	m_yylval = yylval;
}
CToken::CToken(int token)
{
	m_token = token;
}
int CToken::GetToken() const
{
	return m_token;
}
void CToken::GetYYLVAL(YYLVAL & yylval) const
{
	yylval = m_yylval;
}
int CToken::GetPriorityIN() const
{
	if (m_token == GT ||
		m_token == GE ||
		m_token == LE ||
		m_token == LT ||
		m_token == EQ ||
		m_token == NE ||
		m_token == NOT)
	{
		return 1;
	}
	else if (m_token == LBRK)
	{
		return 0;
	}
	else if (m_token == ADD || m_token == SUB)
	{
		return 2;
	}
	else if (m_token == MUL || m_token == DIV)
	{
		return 3;
	}
	else
	{
		fprintf(stderr, "warning:不是操作符，不存在栈内优先级!\n");
		return -1;
	}
}
int CToken::GetPriorityOUT() const
{
	if (m_token == GT ||
		m_token == GE ||
		m_token == LE ||
		m_token == LT ||
		m_token == EQ ||
		m_token == NE ||
		m_token == NOT)
	{
		return 1;
	}
	else if (m_token == ADD || m_token == SUB)
	{
		return 2;
	}
	else if (m_token == MUL || m_token == DIV)
	{
		return 3;
	}
	else
	{
		fprintf(stderr, "warning:不是操作符，不存在栈外优先级!\n");
		return -1;
	}
}
bool CToken::IsExprOperator() const
{
	return (m_token == GT ||
			m_token == GE ||
			m_token == EQ ||
			m_token == NE ||
			m_token == LT ||
			m_token == LE ||
			m_token == ADD||
			m_token == SUB||
			m_token == MUL||
			m_token == DIV||
			m_token == NOT);
}
bool CToken::IsExprOperant() const
{
	return (m_token == VAR ||
			m_token == CONST_FLOAT ||
			m_token == CONST_INT ||
			m_token == CONST_STRING);
}
