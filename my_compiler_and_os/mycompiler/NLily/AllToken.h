// AllToken.h: interface for the AllToken class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ALLTOKEN_H__04D4F36A_7C24_4044_9B6A_60D14468BFE0__INCLUDED_)
#define AFX_ALLTOKEN_H__04D4F36A_7C24_4044_9B6A_60D14468BFE0__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Terminator.h"
#include "NonTerminator.h"

class AllToken  
{
public:
	AllToken();
	virtual ~AllToken();
public:
	//ÖÕ½á·û
	static CTerminator ID;
	static CTerminator CONST_STRING;
	static CTerminator CONST_INTEGER;
	static CTerminator CONST_FLOAT;
	static CTerminator FUNCTION;
	static CTerminator IF;
	static CTerminator THEN;
	static CTerminator ELSE;
	static CTerminator ENDIF;
	static CTerminator WHILE;
	static CTerminator DO;
	static CTerminator ENDWHILE;
	static CTerminator INTEGER;
	static CTerminator STRING;
	static CTerminator FLOAT;
	static CTerminator RETURN;
	static CTerminator BEGIN_FLOW;
	static CTerminator END_FLOW;
	static CTerminator RUN;
	static CTerminator FOR;
	static CTerminator ENDFOR;
	static CTerminator CONTINUE;
	static CTerminator BREAK;
	static CTerminator REPEAT;
	static CTerminator UNTIL;
	static CTerminator SWITCH;
	static CTerminator ENDSWITCH;
	static CTerminator CASE;
	static CTerminator OR;
	static CTerminator AND;
	static CTerminator LT;
	static CTerminator LE;
	static CTerminator EQ;
	static CTerminator NE;
	static CTerminator GT;
	static CTerminator GE;
	static CTerminator UMINUS;
	static CTerminator NOT;
	static CTerminator MEMBLOCK;

	static CTerminator SEMI;
	static CTerminator L_SQ_BRACKET;
	static CTerminator R_SQ_BRACKET;
	static CTerminator ASSIGN;
	static CTerminator ADD;
	static CTerminator SUB;
	static CTerminator MUL;
	static CTerminator DIV;
	static CTerminator MOD;
	static CTerminator L_BRACKET;
	static CTerminator R_BRACKET;
	static CTerminator COMMA;
	static CTerminator COLON;
public:
	//·ÇÖÕ½á·û
	static CNonTerminator flow;
	static CNonTerminator statement_list;
	static CNonTerminator statement;
	static CNonTerminator expr;
	static CNonTerminator declarations;
	static CNonTerminator var_type;
	static CNonTerminator id_list;
	static CNonTerminator flow_code1;
	static CNonTerminator fake1;
	static CNonTerminator fake2;
	static CNonTerminator while_code1;
	static CNonTerminator if_code1;
	static CNonTerminator term;
	static CNonTerminator factor;
	static CNonTerminator function;
	static CNonTerminator arg_list;


};

#endif // !defined(AFX_ALLTOKEN_H__04D4F36A_7C24_4044_9B6A_60D14468BFE0__INCLUDED_)
