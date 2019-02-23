// AllToken.cpp: implementation of the AllToken class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "AllToken.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
//ÖÕ½á·û
CTerminator AllToken::ID("ID");
CTerminator AllToken::CONST_STRING("CONST_STRING");
CTerminator AllToken::CONST_INTEGER("CONST_INTEGER");
CTerminator AllToken::CONST_FLOAT("CONST_FLOAT");
CTerminator AllToken::FUNCTION("FUNCTION");
CTerminator AllToken::IF("IF");
CTerminator AllToken::THEN("THEN");
CTerminator AllToken::ELSE("ELSE");
CTerminator AllToken::ENDIF("ENDIF");
CTerminator AllToken::WHILE("WHILE");
CTerminator AllToken::DO("DO");
CTerminator AllToken::ENDWHILE("ENDWHILE");
CTerminator AllToken::INTEGER("INTEGER");
CTerminator AllToken::STRING("STRING");
CTerminator AllToken::FLOAT("FLOAT");
CTerminator AllToken::RETURN("RETURN");
CTerminator AllToken::BEGIN_FLOW("BEGIN_FLOW");
CTerminator AllToken::END_FLOW("END_FLOW");
CTerminator AllToken::RUN("RUN");
CTerminator AllToken::FOR("FOR");
CTerminator AllToken::ENDFOR("ENDFOR");
CTerminator AllToken::CONTINUE("CONTINUE");
CTerminator AllToken::BREAK("BREAK");
CTerminator AllToken::REPEAT("REPEAT");
CTerminator AllToken::UNTIL("UNTIL");
CTerminator AllToken::SWITCH("SWITCH");
CTerminator AllToken::ENDSWITCH("ENDSWITCH");
CTerminator AllToken::CASE("CASE");
CTerminator AllToken::OR("OR");
CTerminator AllToken::AND("AND");
CTerminator AllToken::LT("LT");
CTerminator AllToken::LE("LE");
CTerminator AllToken::EQ("EQ");
CTerminator AllToken::NE("NE");
CTerminator AllToken::GT("GT");
CTerminator AllToken::GE("GE");
CTerminator AllToken::UMINUS("UMINUS");
CTerminator AllToken::NOT("NOT");
CTerminator AllToken::MEMBLOCK("MEMBLOCK");

CTerminator AllToken::SEMI(";");
CTerminator AllToken::L_SQ_BRACKET("[");
CTerminator AllToken::R_SQ_BRACKET("]");
CTerminator AllToken::ASSIGN("=");
CTerminator AllToken::ADD("+");
CTerminator AllToken::SUB("-");
CTerminator AllToken::MUL("*");
CTerminator AllToken::DIV("/");
CTerminator AllToken::MOD("%");
CTerminator AllToken::L_BRACKET("(");
CTerminator AllToken::R_BRACKET(")");
CTerminator AllToken::COMMA(",");
CTerminator AllToken::COLON(":");

//·ÇÖÕ½á·ûºÅ
CNonTerminator AllToken::flow("flow");
CNonTerminator AllToken::statement_list("statement_list");
CNonTerminator AllToken::statement("statement");
CNonTerminator AllToken::expr("expr");
CNonTerminator AllToken::declarations("declarations");
CNonTerminator AllToken::var_type("var_type");
CNonTerminator AllToken::id_list("id_list");
CNonTerminator AllToken::flow_code1("flow_code1");
CNonTerminator AllToken::fake1("fake1");
CNonTerminator AllToken::fake2("fake2");
CNonTerminator AllToken::while_code1("while_code1");
CNonTerminator AllToken::if_code1("if_code1");
CNonTerminator AllToken::term("term");
CNonTerminator AllToken::factor("factor");
CNonTerminator AllToken::function("function");
CNonTerminator AllToken::arg_list("arg_list");



AllToken::AllToken()
{

}

AllToken::~AllToken()
{

}
