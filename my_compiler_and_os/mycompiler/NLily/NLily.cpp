// NLily.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Dot.h"
#include "Terminator.h"
#include "NonTerminator.h"
#include "symbol_list.h"
#include "Producer.h"
#include "Producer_Set.h"
#include "Item_Set.h"
#include "Grammar.h"
#include "items.h"
#include "AllToken.h"

#include "Lex.h"


int main(int argc, char* argv[])
{
//#define __TEST
#ifndef __TEST
	CSymbol_List list;
	CProducer_Set G;

	CNonTerminator S("S");

	//S->flow
	list.clear();
	list.push_back(&AllToken::flow);
	{
		CProducer prd(S, list, NULL);
		G.insert(&prd);
	}

	//flow -> L_SQ_BRACKET ID R_SQ_BRACKET BEGIN_FLOW flow_code1 statement_list END_FLOW
	list.clear();
	list.push_back(&AllToken::L_SQ_BRACKET);
	list.push_back(&AllToken::ID);
	list.push_back(&AllToken::R_SQ_BRACKET);
	list.push_back(&AllToken::BEGIN_FLOW);
	list.push_back(&AllToken::flow_code1);
	list.push_back(&AllToken::statement_list);
	list.push_back(&AllToken::END_FLOW);
	{
		CProducer prd(AllToken::flow, list, NULL);
		G.insert(&prd);
	}
	//伪非终结符号，仅用来产生一段代码
	//flow_code1->EPSL
	list.clear();
	list.push_back(&CTerminator::EPSL);
	{
		CProducer prd(AllToken::flow_code1, list, NULL);
		G.insert(&prd);
	}
	//statement_list->statement_list statement
	list.clear();
	list.push_back(&AllToken::statement_list);
	list.push_back(&AllToken::statement);
	{
		CProducer prd(AllToken::statement_list, list, NULL);
		G.insert(&prd);
	}	
	//statement_list->statement
	list.clear();
	list.push_back(&AllToken::statement);
	{
		CProducer prd(AllToken::statement_list, list, NULL);
		G.insert(&prd);
	}	
	//statement->expr SEMI
	list.clear();
	list.push_back(&AllToken::expr);
	list.push_back(&AllToken::SEMI);
	{
		CProducer prd(AllToken::statement, list, NULL);
		G.insert(&prd);
	}
	//statement->ID ASSIGN expr SEMI 赋值语句
	list.clear();
	list.push_back(&AllToken::ID);
	list.push_back(&AllToken::ASSIGN);
	list.push_back(&AllToken::expr);
	list.push_back(&AllToken::SEMI);
	{
		CProducer prd(AllToken::statement, list, NULL);
		G.insert(&prd);
	}
	//statement->declarations
	list.clear();
	list.push_back(&AllToken::declarations);
	{
		CProducer prd(AllToken::statement, list, NULL);
		G.insert(&prd);
	}
	//statement->IF L_BRACKET expr R_BRACKET fake1 THEN statement_list ENDIF
	list.clear();
	list.push_back(&AllToken::IF);
	list.push_back(&AllToken::L_BRACKET);
	list.push_back(&AllToken::expr);
	list.push_back(&AllToken::R_BRACKET);
	list.push_back(&AllToken::fake1);
	list.push_back(&AllToken::THEN);
	list.push_back(&AllToken::statement_list);
	list.push_back(&AllToken::ENDIF);
	{
		CProducer prd(AllToken::statement, list, NULL);
		G.insert(&prd);
	}
	//fake1->EPSL
	list.clear();
	list.push_back(&CTerminator::EPSL);
	{
		CProducer prd(AllToken::fake1, list, NULL);
		G.insert(&prd);
	}
	////////////////////////////////////////////////
	//statement->IF L_BRACKET expr R_BRACKET fake1 THEN statement_list 
	//ELSE  if_code1 statement_list ENDIF
	list.clear();
	list.push_back(&AllToken::IF);
	list.push_back(&AllToken::L_BRACKET);
	list.push_back(&AllToken::expr);
	list.push_back(&AllToken::R_BRACKET);
	list.push_back(&AllToken::fake1);
	list.push_back(&AllToken::THEN);
	list.push_back(&AllToken::statement_list);
	list.push_back(&AllToken::ELSE);
	list.push_back(&AllToken::if_code1);
	list.push_back(&AllToken::statement_list);
	list.push_back(&AllToken::ENDIF);
	{
		CProducer prd(AllToken::statement, list, NULL);
		G.insert(&prd);
	}
	//if_code1->EPSL
	list.clear();
	list.push_back(&CTerminator::EPSL);
	{
		CProducer prd(AllToken::if_code1, list, NULL);
		G.insert(&prd);
	}
	////////////////////////////////////////////////
	//statement->WHILE fake2 L_BRACKET expr  R_BRACKET while_code1 DO statement_list ENDWHILE
	list.clear();
	list.push_back(&AllToken::WHILE);
	list.push_back(&AllToken::fake2);
	list.push_back(&AllToken::L_BRACKET);
	list.push_back(&AllToken::expr);
	list.push_back(&AllToken::R_BRACKET);
	list.push_back(&AllToken::while_code1);
	list.push_back(&AllToken::DO);
	list.push_back(&AllToken::statement_list);
	list.push_back(&AllToken::ENDWHILE);
	{
		CProducer prd(AllToken::statement, list, NULL);
		G.insert(&prd);
	}
	//fake2->EPSL
	list.clear();
	list.push_back(&CTerminator::EPSL);
	{
		CProducer prd(AllToken::fake2, list, NULL);
		G.insert(&prd);
	}
	//while_code1->EPSL
	list.clear();
	list.push_back(&CTerminator::EPSL);
	{
		CProducer prd(AllToken::while_code1, list, NULL);
		G.insert(&prd);
	}
	////////////////////////////////////////////////////
	//statement->RUN  L_BRACKET expr R_BRACKET SEMI
	list.clear();
	list.push_back(&AllToken::RUN);
	list.push_back(&AllToken::L_BRACKET);
	list.push_back(&AllToken::expr);
	list.push_back(&AllToken::R_BRACKET);
	list.push_back(&AllToken::SEMI);
	{
		CProducer prd(AllToken::statement, list, NULL);
		G.insert(&prd);
	}
	//statement->RETURN SEMI
	list.clear();
	list.push_back(&AllToken::RETURN);
	list.push_back(&AllToken::SEMI);
	{
		CProducer prd(AllToken::statement, list, NULL);
		G.insert(&prd);
	}
	//statement->BREAK SEMI
	list.clear();
	list.push_back(&AllToken::BREAK);
	list.push_back(&AllToken::SEMI);
	{
		CProducer prd(AllToken::statement, list, NULL);
		G.insert(&prd);
	}
	//statement->CONTINUE SEMI
	list.clear();
	list.push_back(&AllToken::CONTINUE);
	list.push_back(&AllToken::SEMI);
	{
		CProducer prd(AllToken::statement, list, NULL);
		G.insert(&prd);
	}
	//statement->SEMI
	list.clear();
	list.push_back(&AllToken::SEMI);
	{
		CProducer prd(AllToken::statement, list, NULL);
		G.insert(&prd);
	}
	//declarations->var_type id_list ;
	list.clear();
	list.push_back(&AllToken::var_type);
	list.push_back(&AllToken::id_list);
	list.push_back(&AllToken::SEMI);
	{
		CProducer prd(AllToken::declarations, list, NULL);
		G.insert(&prd);
	}
	//var_type->INTEGER
	list.clear();
	list.push_back(&AllToken::INTEGER);
	{
		CProducer prd(AllToken::var_type, list, NULL);
		G.insert(&prd);
	}
	//var_type->FLOAT
	list.clear();
	list.push_back(&AllToken::FLOAT);
	{
		CProducer prd(AllToken::var_type, list, NULL);
		G.insert(&prd);
	}
	//var_type->STRING
	list.clear();
	list.push_back(&AllToken::STRING);
	{
		CProducer prd(AllToken::var_type, list, NULL);
		G.insert(&prd);
	}
	//id_list->id_list , ID
	list.clear();
	list.push_back(&AllToken::id_list);
	list.push_back(&AllToken::COMMA);
	list.push_back(&AllToken::ID);
	{
		CProducer prd(AllToken::id_list, list, NULL);
		G.insert(&prd);
	}
	//id_list->ID
	list.clear();
	list.push_back(&AllToken::ID);
	{
		CProducer prd(AllToken::id_list, list, NULL);
		G.insert(&prd);
	}
	//////////////////一/////////////////////////////
	//expr -> expr ADD term
	list.clear();
	list.push_back(&AllToken::expr);
	list.push_back(&AllToken::ADD);
	list.push_back(&AllToken::term);
	{
		CProducer prd(AllToken::expr, list, NULL);
		G.insert(&prd);
	}	
	//expr -> expr OR term
	list.clear();
	list.push_back(&AllToken::expr);
	list.push_back(&AllToken::OR);
	list.push_back(&AllToken::term);
	{
		CProducer prd(AllToken::expr, list, NULL);
		G.insert(&prd);
	}	
	//expr -> expr SUB term
	list.clear();
	list.push_back(&AllToken::expr);
	list.push_back(&AllToken::SUB);
	list.push_back(&AllToken::term);
	{
		CProducer prd(AllToken::expr, list, NULL);
		G.insert(&prd);
	}
	//expr ->term
	list.clear();
	list.push_back(&AllToken::term);
	{
		CProducer prd(AllToken::expr, list, NULL);
		G.insert(&prd);
	}
/////////////////////二////////////////////////////////
	//term->term MUL factor
	list.clear();
	list.push_back(&AllToken::term);
	list.push_back(&AllToken::MUL);
	list.push_back(&AllToken::factor);
	{
		CProducer prd(AllToken::term, list, NULL);
		G.insert(&prd);
	}
	//term->term DIV factor
	list.clear();
	list.push_back(&AllToken::term);
	list.push_back(&AllToken::DIV);
	list.push_back(&AllToken::factor);
	{
		CProducer prd(AllToken::term, list, NULL);
		G.insert(&prd);
	}
	//term->term MOD factor
	list.clear();
	list.push_back(&AllToken::term);
	list.push_back(&AllToken::MOD);
	list.push_back(&AllToken::factor);
	{
		CProducer prd(AllToken::term, list, NULL);
		G.insert(&prd);
	}
	//term->term AND factor
	list.clear();
	list.push_back(&AllToken::term);
	list.push_back(&AllToken::AND);
	list.push_back(&AllToken::factor);
	{
		CProducer prd(AllToken::term, list, NULL);
		G.insert(&prd);
	}
	//term->term LT factor
	list.clear();
	list.push_back(&AllToken::term);
	list.push_back(&AllToken::LT);
	list.push_back(&AllToken::factor);
	{
		CProducer prd(AllToken::term, list, NULL);
		G.insert(&prd);
	}
	//term->term LE factor
	list.clear();
	list.push_back(&AllToken::term);
	list.push_back(&AllToken::LE);
	list.push_back(&AllToken::factor);
	{
		CProducer prd(AllToken::term, list, NULL);
		G.insert(&prd);
	}
	//term->term GT factor
	list.clear();
	list.push_back(&AllToken::term);
	list.push_back(&AllToken::GT);
	list.push_back(&AllToken::factor);
	{
		CProducer prd(AllToken::term, list, NULL);
		G.insert(&prd);
	}
	//term->term GE factor
	list.clear();
	list.push_back(&AllToken::term);
	list.push_back(&AllToken::GE);
	list.push_back(&AllToken::factor);
	{
		CProducer prd(AllToken::term, list, NULL);
		G.insert(&prd);
	}
	//term->term EQ factor
	list.clear();
	list.push_back(&AllToken::term);
	list.push_back(&AllToken::EQ);
	list.push_back(&AllToken::factor);
	{
		CProducer prd(AllToken::term, list, NULL);
		G.insert(&prd);
	}
	//term->term NE factor
	list.clear();
	list.push_back(&AllToken::term);
	list.push_back(&AllToken::EQ);
	list.push_back(&AllToken::factor);
	{
		CProducer prd(AllToken::term, list, NULL);
		G.insert(&prd);
	}
	//term->factor
	list.clear();
	list.push_back(&AllToken::factor);
	{
		CProducer prd(AllToken::term, list, NULL);
		G.insert(&prd);
	}
	//////////////////////三////////////////
	//factor->ID
	list.clear();
	list.push_back(&AllToken::ID);
	{
		CProducer prd(AllToken::factor, list, NULL);
		G.insert(&prd);
	}
	
	//factor->L_BRACKET expr R_BRACKET
	list.clear();
	list.push_back(&AllToken::L_BRACKET);
	list.push_back(&AllToken::expr);
	list.push_back(&AllToken::R_BRACKET);
	{
		CProducer prd(AllToken::factor, list, NULL);
		G.insert(&prd);
	}

	//factor->CONST_STRING
	list.clear();
	list.push_back(&AllToken::CONST_STRING);
	{
		CProducer prd(AllToken::factor, list, NULL);
		G.insert(&prd);
	}
	//factor->CONST_FLOAT
	list.clear();
	list.push_back(&AllToken::CONST_FLOAT);
	{
		CProducer prd(AllToken::factor, list, NULL);
		G.insert(&prd);
	}
	//factor->CONST_INTEGER
	list.clear();
	list.push_back(&AllToken::CONST_INTEGER);
	{
		CProducer prd(AllToken::factor, list, NULL);
		G.insert(&prd);
	}
	//factor->function
	list.clear();
	list.push_back(&AllToken::function);
	{
		CProducer prd(AllToken::factor, list, NULL);
		G.insert(&prd);
	}
	//function->ID L_BRACKET arg_list R_BRACKET
	list.clear();
	list.push_back(&AllToken::ID);
	list.push_back(&AllToken::L_BRACKET);
	list.push_back(&AllToken::arg_list);
	list.push_back(&AllToken::R_BRACKET);
	{
		CProducer prd(AllToken::function, list, NULL);
		G.insert(&prd);
	}
	//arg_list->arg_list COMMA expr
	list.clear();
	list.push_back(&AllToken::arg_list);
	list.push_back(&AllToken::COMMA);
	list.push_back(&AllToken::expr);
	{
		CProducer prd(AllToken::arg_list, list, NULL);
		G.insert(&prd);
	}
	//arg_list->expr
	list.clear();
	list.push_back(&AllToken::expr);
	{
		CProducer prd(AllToken::arg_list, list, NULL);
		G.insert(&prd);
	}
	//arg_list->EPSL
	list.clear();
	list.push_back(&CTerminator::EPSL);
	{
		CProducer prd(AllToken::arg_list, list, NULL);
		G.insert(&prd);
	}
	//factor->SUB expr
	list.clear();
	list.push_back(&AllToken::SUB);
	list.push_back(&AllToken::expr);
	{
		CProducer prd(AllToken::factor, list, NULL);
		G.insert(&prd);
	}
	//factor->NOT expr
	list.clear();
	list.push_back(&AllToken::NOT);
	list.push_back(&AllToken::expr);
	{
		CProducer prd(AllToken::factor, list, NULL);
		G.insert(&prd);
	}

	CGrammar ggg(S,  G);

	if (CLex::init_lex("e:\\aaa.txt") < 0)
	{
		fprintf(stderr, "输入文件打开失败!\n");
		return -1;
	}

	ggg.CalculateAnalyseTable();
//	ggg.ReadAnalyseTableFrmFile("e:\\analyse.txt");
	ggg.WriteAnalyseTableToFile("e:\\analyse.txt");
	printf("<<<%d\n", ggg.parse());

	while (1)
	{
		CTerminator ttt = ggg.lex();
		if (ttt == CTerminator::FINIS)
		{
			break;
		}
		printf("%s\n", ttt.ToString().c_str());
	}

#else

	if (CLex::init_lex("e:\\aaa.txt") < 0)
	{
		fprintf(stderr, "输入文件打开失败!\n");
		return -1;
	}
	int ret;
	while ( (ret = CLex::lex()) != 0)
	{
		printf("%d %s\n", ret, CLex::getyytext());
	}
#endif
	return 0;
}

