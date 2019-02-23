// Grammar.cpp: implementation of the CGrammar class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Grammar.h"
#include <assert.h>
#include "StateOrSymbol.h"
#include "StateOrSymbol_Stack.h"
#include "Terminator_Index.h"
#include "AllToken.h"
#include "Lex.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CGrammar::CGrammar(const CNonTerminator& StartSymbol,
			const CProducer_Set& PS )
	:m_StartSymbol(StartSymbol), m_PS(PS)
{
	m_SL.clear();
	m_PS.begin_iterator();
	const CProducer * pprd = NULL;
	while ( (pprd = m_PS.next()) != NULL)
	{
		const CSymbol* psym = &(pprd->GetLeft());
		if (!m_SL.contain(psym))
		{
			m_SL.push_back(psym);
		}
		CSymbol_List right = pprd->GetRight();
		right.begin_iterator();
		while ( (psym = right.next()) != NULL)
		{
			if (!m_SL.contain(psym))
			{
				m_SL.push_back(psym);
			}
		}
	}
#ifdef _DEBUG
	{
		m_SL.begin_iterator();
		const CSymbol* ppp = NULL;
		while ( (ppp = m_SL.next()) != NULL)
		{
			printf(">>>>%s\n", ppp->ToString().c_str());
		}
	}
#endif
}

CGrammar::~CGrammar()
{

}

CSymbol_List CGrammar::FIRST(const CSymbol *X, CProducer_Set G)
{
#ifdef _DEBUG
	//printf("--FIRST CALL %s\n", X->ToString().c_str());
#endif
	
	CSymbol_List ret;

	assert(X->GetSymType() != SYMBOL_DOT);
	if (X->GetSymType() == SYMBOL_TERMINATOR)
	{
		ret.push_back(X);
		return ret;
	}

	//如果X->ε是一个产生式，将ε加入FIRST(X)
	CSymbol_List right;
	right.push_back(&CTerminator::EPSL);
	CProducer prd(*(CNonTerminator*)X, right, NULL);
	if (G.contain(prd))//如果prd是一个产生式
	{
		if (!ret.contain(&(CTerminator::EPSL)))
		{
			ret.push_back(&CTerminator::EPSL);
		}
	}
	
	G.begin_iterator();
	const CProducer * pprd = NULL;
	while ( (pprd = G.next()) != NULL)
	{
		
		if (! (pprd->GetLeft() == (*(CNonTerminator*)X) ) )
		{
			continue;
		}
		// a) FIRST(Y1)的所有符号都在FIRST(X)中
		right = pprd->GetRight();
		right.begin_iterator();
		const CSymbol* Y1 = right.next();

		///权宜加上这个判断
		if (Y1->GetSymType() == SYMBOL_NONTERMINATOR)
		{
			if ( (*(CNonTerminator*)Y1) == (*(CNonTerminator*)X) )
			{
				continue;
			}
		}


		CSymbol_List firstY1 = CGrammar::FIRST(Y1, G);
		firstY1.begin_iterator();
		const CSymbol* symbInFirstY1 = NULL;
		while ( (symbInFirstY1 = firstY1.next()) != NULL)
		{
			if (!ret.contain(symbInFirstY1))
			{
				ret.push_back(symbInFirstY1);
			}
		}
	
		// b)若对于某个i, a属于FIRST(Yi)且ε属于
		//FIRST(Y1),...,FIRST(Yi-1),则将a加入FIRST(X)
		right = pprd->GetRight();
		right.begin_iterator();
		const CSymbol * Yi = NULL;
		bool bAllHasEPSL = TRUE;//right的每个符号的first集都含有ε
		while ( (Yi = right.next()) != NULL)
		{
			CSymbol_List firstYi = FIRST(Yi, G);
			if (firstYi.contain(&CTerminator::EPSL))
			{
				continue;
			}
			bAllHasEPSL = FALSE;
			const CSymbol* symbolInFirstYi = NULL;
			firstYi.begin_iterator();
			while ( (symbolInFirstYi = firstYi.next()) != NULL)
			{
				if ( !ret.contain(symbolInFirstYi))
				{
					ret.push_back(symbolInFirstYi);
				}
			}
			break;//再也不需要往下走了
		}
		// c)如果对所有的j=1,2,...,k, ε在FIRST(Yk)中，
		//那么将ε加到FIRST(X)中
		if (bAllHasEPSL)
		{
			if (!ret.contain(&CTerminator::EPSL))
			{
				ret.push_back(&CTerminator::EPSL);
			}
		}

	}
	return ret;
}
/*
*求项目集I的闭包，G是文法的产生式的集合, GG是与文法等价的非递归文法的产生式的集合
*/
CItem_Set CGrammar::Closure(const CItem_Set I, const CProducer_Set G, const CProducer_Set &GG)
{
	/*
	CItem_Set ret = I;
	
	bool bNewItemAdded;
	do
	{
		bNewItemAdded = FALSE;

		const CItem_Set J = ret;
		J.begin_iterator();
		const CItem * item;
		while ( (item = J.next()) != NULL)//每个[A->α.Bβ,a]项目
		{
			int dotIndex = item->GetProducer().GetDotIndex();
			const CSymbol * B = item->GetProducer().GetSymbolAt(dotIndex + 1);
			if (NULL == B)
			{
				continue;
			}
			//求FIRST(βa)
			CSymbol_List first_Beta_a;
			CSymbol_List beta;
			//先得到beta
			int iii = 0;
			const CSymbol * ppp = NULL;
			while ( (ppp = item->GetProducer().GetSymbolAt(dotIndex + 2 + iii)) != NULL)
			{
				beta.push_back(ppp);
				++iii;
			}
			CSymbol_List beta_a = beta;
			CTerminator a(item->GetTerminator());
			beta_a.push_back(&a);
			first_Beta_a = CGrammar::FIRST(beta_a, GG);
		

			G.begin_iterator();
			const CProducer * pprd = NULL;
			while ( (pprd = G.next()) != NULL)
			{
				if ( !(pprd->GetLeft() == (*(CNonTerminator*)B) ) )//产生式左部不为B
				{
					continue;
				}
				//构造[B->.γ, b]项目
				CSymbol_List garma = pprd->GetRight();
				garma.push_front(&(CDot()));

				first_Beta_a.begin_iterator();
				const CSymbol * b = NULL;
				while ( (b = first_Beta_a.next()) != NULL)
				{
					CProducer newprd(*(CNonTerminator*)B, garma, NULL);
					CItem newitem(newprd, *(CTerminator*)b);
					if (!ret.contain(newitem))
					{
						ret.insert(&newitem);
						bNewItemAdded = TRUE;
					}
				}
			}
		}
	}
	while (bNewItemAdded);
	return ret;
	*/
	CItem_Set ret = I;
	CItem_Set incr_set = I;
	
	bool bNewItemAdded;
	do
	{
		bNewItemAdded = FALSE;

		const CItem_Set J = incr_set;
		incr_set.clear();
		J.begin_iterator();
		const CItem * item;
		while ( (item = J.next()) != NULL)//每个项目[A->α.Bβ,a]
		{
			int dotIndex = item->GetProducer().GetDotIndex();
			const CSymbol * B = item->GetProducer().GetSymbolAt(dotIndex + 1);
			if (NULL == B)
			{
				continue;
			}
			//求FIRST(βa)
			CSymbol_List first_Beta_a;
			CSymbol_List beta;
			//先得到beta
			int iii = 0;
			const CSymbol * ppp = NULL;
			while ( (ppp = item->GetProducer().GetSymbolAt(dotIndex + 2 + iii)) != NULL)
			{
				beta.push_back(ppp);
				++iii;
			}
			CSymbol_List beta_a = beta;
			CTerminator a(item->GetTerminator());
			beta_a.push_back(&a);
			first_Beta_a = CGrammar::FIRST(beta_a, GG);
		

			G.begin_iterator();
			const CProducer * pprd = NULL;
			while ( (pprd = G.next()) != NULL)//每个产生式
			{
				if ( !(pprd->GetLeft() == (*(CNonTerminator*)B) ) )//产生式左部不为B
				{
					continue;
				}
				//构造[B->.γ, b]项目
				CSymbol_List garma = pprd->GetRight();
				garma.push_front(&(CDot()));

				first_Beta_a.begin_iterator();
				const CSymbol * b = NULL;
				while ( (b = first_Beta_a.next()) != NULL)//每个终结符号
				{
					CProducer newprd(*(CNonTerminator*)B, garma, NULL);
					CItem newitem(newprd, *(CTerminator*)b);
					if (!ret.contain(newitem))
					{
						ret.insert(&newitem);
						incr_set.insert(&newitem);
						bNewItemAdded = TRUE;
					}
				}
			}
		}
	}
	while (bNewItemAdded);
	return ret;
}

CItem_Set CGrammar::Goto(const CItem_Set I, const CSymbol *X, const CProducer_Set G, const CProducer_Set& GG)
{
	assert(NULL != X);
	assert(X->GetSymType() != SYMBOL_DOT);

	CItem_Set J;

	I.begin_iterator();
	const CItem * pitem = NULL;
	while ( (pitem = I.next()) != NULL)
	{
		int dotIndex = pitem->GetProducer().GetDotIndex();
		const CSymbol * pX = pitem->GetProducer().GetSymbolAt(dotIndex + 1);
		if (NULL == pX)
		{
			continue;
		}
		if (pX->GetSymType() != X->GetSymType())
		{
			continue;
		}
		if (pX->GetSymType() == SYMBOL_TERMINATOR)
		{
			if ( !((*(CTerminator*)pX) == (*(CTerminator*)X)) )
			{
				continue;
			}
		}
		else if (pX->GetSymType() == SYMBOL_NONTERMINATOR)
		{
			if ( !((*(CNonTerminator*)pX) == (*(CNonTerminator*)X)) )
			{
				continue;
			}
		}
		//到这里pX确实就等于X
		CSymbol_List right = pitem->GetProducer().GetRight();
		right.insert_before(&CDot(), dotIndex + 2);
		right.removeAt(dotIndex);
		CProducer newprd(pitem->GetProducer().GetLeft(), right, pitem->GetProducer().GetFunc());
		CItem newitem(newprd, pitem->GetTerminator());
		J.insert(&newitem);
	}
	return Closure(J, G, GG);
}
/*
*	C:项目集的集合
*	I0:初始项目集
*	SL:文法中所有文法符号的集合
*	G:文法中所有产生式的集合
*/
void CGrammar::Items(items &C, const CItem_Set I0, const CSymbol_List SL, const CProducer_Set G, const CProducer_Set& GG)
{
	/*
	C.clear();
	C.add(I0);
	int count = 0;
	
	bool bNewEleAdded;
	do
	{
		bNewEleAdded = FALSE;
		C.begin_iterator();
		const CItem_Set * I = NULL;
		while ( ( I = C.next()) != NULL)//每个项目集
		{
			SL.begin_iterator();
			const CSymbol* X = NULL;
			while ( (X = SL.next()) !=  NULL)//每个文法符号
			{
				CItem_Set newitemset = CGrammar::Goto(*I, X, G, GG);
				if (newitemset.size() == 0)//为空
				{
					continue;
				}
				if ( C.contain(newitemset) )//已经存在C中
				{
					continue;
				}
				C.add(newitemset);
				bNewEleAdded = TRUE;
				printf("----------增加一个项目集%d\n", ++count);
			}
		}
	}
	while (bNewEleAdded);
	*/
	C.clear();
	C.add(I0);
	int count = 0;
	
	bool bNewEleAdded;
	items ttt = C;
	do
	{
		bNewEleAdded = FALSE;
		const items jjj = ttt;
		ttt.clear();
		jjj.begin_iterator();
		const CItem_Set * I = NULL;


		printf("-----jjj.size()=%d\n", jjj.size());

	
		while ( ( I = jjj.next()) != NULL)//每个项目集
		{
	
			SL.begin_iterator();
			const CSymbol* X = NULL;
			while ( (X = SL.next()) !=  NULL)//每个文法符号
			{
				CItem_Set newitemset = CGrammar::Goto(*I, X, G, GG);
				if (newitemset.size() == 0)//为空
				{
					continue;
				}
				if ( C.contain(newitemset) )//已经存在C中
				{
					continue;
				}
				C.add(newitemset);
				ttt.add(newitemset);
				bNewEleAdded = TRUE;
				printf("----------增加一个项目集%d\n", ++count);
			}
		}
	}
	while (bNewEleAdded);
}

/*
*table:分析表
*
*/

void CGrammar::CalculateAnalyseTable(CAnalyseTable& table, const CSymbol_List SL, const CProducer_Set G, const CItem_Set I0, const CNonTerminator  StartSymbol, const CProducer_Set& GG)
{
	table.clear();
	//1.构造项目集规范族
	items C;
	printf("----------开始构造项目集规范族\n");
	CGrammar::Items(C, I0, SL, G, GG);
	printf("----------构造项目集规范族完毕\n");


	//2.状态的动作
	printf("----------开始计算ACTION\n");
	C.begin_iterator();
	const CItem_Set * pItemSet = NULL;
	while ( (pItemSet = C.next()) != NULL)
	{
		
		CItem_Set Ii(*pItemSet);
		const int i = C.GetItemSetIndex(Ii);
		Ii.begin_iterator();
		const CItem* pItem = NULL;
		while ( (pItem = Ii.next()) != NULL)
		{
			
			const int dotIndex = pItem->GetProducer().GetDotIndex();
			const CSymbol* a = pItem->GetProducer().GetSymbolAt(dotIndex + 1);
			//c)
			if ( (pItem->GetProducer().GetLeft()) == StartSymbol &&
				(dotIndex + 1) == (pItem->GetProducer().GetRight().size()))
			{
				ACTION newaction, oldaction;

				newaction.state = i;
				newaction.terminator = CTerminator::FINIS;
				sprintf(newaction.action, "acc");
				
				oldaction.state = i;
				oldaction.terminator = CTerminator::FINIS;
				
				if ( table.GetAction(oldaction) == 0 )
				{
					if (strcmp(oldaction.action, newaction.action) != 0)
					{
						fprintf(stderr, "[%s][%d]文法不是LR(1)文法,构造分析表冲突!\n",
							__FILE__,
							__LINE__);
						exit(-1);
					}
				}
				else
				{
					table.AddAction(newaction);
				}
				continue;
			}
			//a)
			if ( (NULL != a) && 
				 ( (a->GetSymType()) == SYMBOL_TERMINATOR) )
			{
				CItem_Set Ij = CGrammar::Goto(Ii, a, G, GG);
				const int j = C.GetItemSetIndex(Ij);

				ACTION newaction, oldaction;

				newaction.state = i;
				newaction.terminator = *(CTerminator*)a;
				sprintf(newaction.action, "s%d", j);

				oldaction.state = i;
				oldaction.terminator = *(CTerminator*)a;

				if ( table.GetAction(oldaction) == 0 )
				{
					if (strcmp(oldaction.action, newaction.action) != 0)
					{
						fprintf(stderr, "[%s][%d]文法不是LR(1)文法,构造分析表冲突!\n",
							__FILE__,
							__LINE__);
						exit(-1);
					}
				}
				else
				{
					table.AddAction(newaction);
				}
				continue;
			}
			//b)
			if (a == NULL)
			{
				CTerminator b = pItem->GetTerminator();
				CSymbol_List right = pItem->GetProducer().GetRight();
				right.removeAt(dotIndex);
				CNonTerminator A = pItem->GetProducer().GetLeft();
				CProducer newprd(A, right, NULL);
				const int j = G.GetProducerIndex(newprd);
				if (j < 0)
				{
					fprintf(stderr, "[%s][%d]产生式无效!\n",
						__FILE__,
						__LINE__);
					exit(-1);
				}
				ACTION newaction, oldaction;

				newaction.state = i;
				newaction.terminator = b;
				sprintf(newaction.action, "r%d", j);
		
				oldaction.state = i;
				oldaction.terminator = b;

				if ( table.GetAction(oldaction) == 0)
				{
					if (strcmp(oldaction.action, newaction.action) != 0)
					{
						fprintf(stderr, "[%s][%d]文法不是LR(1)文法,构造分析表冲突!\n",
							__FILE__,
							__LINE__);
						exit(-1);
					}
				}
				else
				{
					table.AddAction(newaction);
				}
				continue;
			}
		
		}
	}
	printf("----------计算ACTION结束\n");
	
	//3.状态的转移
	printf("----------开始计算GOTO\n");
	C.begin_iterator();
	pItemSet = NULL;
	while ( (pItemSet = C.next()) != NULL)
	{
		
		CItem_Set Ii(*pItemSet);
		const int i = C.GetItemSetIndex(Ii);
		
		SL.begin_iterator();
		const CSymbol * psym = NULL;
		while ( (psym = SL.next()) != NULL )
		{
			if (psym->GetSymType() != SYMBOL_NONTERMINATOR)
			{
				continue;
			}

			CItem_Set Ij = CGrammar::Goto(Ii, psym, G, GG);
			const int j = C.GetItemSetIndex(Ij);
			if ( j < 0)
			{
				continue;
			}

			GOTO ggg, oldggg;
			ggg.state = i;
			ggg.nonterminator = *(CNonTerminator*)psym;
			ggg.gotostate = j;
			oldggg.nonterminator = *(CNonTerminator*)psym;
			oldggg.state = i;
			if (table.GetGoto(oldggg) == 0)
			{
				if (oldggg.gotostate != ggg.gotostate)
				{
					fprintf(stderr, "[%s][%d]文法不是LR(1)文法!\n",
							__FILE__,
							__LINE__);
					exit(-1);
				}
			}
			else
			{
				table.AddGoto(ggg);
			}
	
		}
	}
	printf("----------计算GOTO结束\n");
}

CGrammar::CGrammar(const CGrammar &g)
:m_StartSymbol(g.m_StartSymbol), m_SL(g.m_SL), m_PS(g.m_PS), m_table(g.m_table)
{

}

const CGrammar& CGrammar::operator =(const CGrammar &another)
{
	m_StartSymbol = another.m_StartSymbol;
	m_SL = another.m_SL;
	m_PS = another.m_PS;
	return *this;
}

bool CGrammar::operator ==(const CGrammar &another) const
{
	return (m_StartSymbol == another.m_StartSymbol &&
			m_SL == another.m_SL &&
			m_PS == another.m_PS);
}
/*
*返回与该文法等价的无左递归文法
*/
CGrammar CGrammar::ClearRecursion() const
{
	CGrammar ret = *this;

	///////////////////////////////////
	//算法4.1 消除间接递归
	//1.以某种顺序排列非终结符号A1,A2,...,An
	CNonTerminator* A = NULL;
	int A_SIZE = 0;
	{
		CSymbol_List list;
		const CSymbol * psym = NULL;
		m_SL.begin_iterator();
		while ( (psym = m_SL.next()) != NULL)
		{
			if (psym->GetSymType() == SYMBOL_NONTERMINATOR)
			{
				list.push_back(psym);
			}
		}
		A_SIZE = list.size();
		A = new CNonTerminator[A_SIZE];
		if (NULL == A)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
				__FILE__,
				__LINE__);
			exit(-1);
		}
		int i;
		list.begin_iterator();
		for (i = 0; i < list.size(); i++)
		{
			A[i] = *(CNonTerminator*)(list.next());
		}
	}
	// 2....
	int i, j;
	for (i = 0; i < A_SIZE; ++i)
	{
		CNonTerminator Ai = A[i];
		for (j = 0; j < i; ++j)
		{
			CNonTerminator Aj = A[j];
			CProducer_Set Aj_prdset = m_PS.GetProducerOfNonTerm(Aj);
			
			CProducer_Set Ai_prdset = ret.m_PS.RmAiAjProducer(Ai, Aj);//删除形如Ai->Aj γ的产生式

			Ai_prdset.begin_iterator();
			const CProducer * Ai_pprd = NULL;
			while ( (Ai_pprd = Ai_prdset.next()) != NULL)
			{
				CSymbol_List r = Ai_pprd->GetRight();
				r.removeAt(0);

				Aj_prdset.begin_iterator();
				const CProducer * Aj_pprd = NULL;
				while ( (Aj_pprd = Aj_prdset.next()) != NULL)
				{
					CSymbol_List q = Aj_pprd->GetRight();
					q.addAll(r);
					q.trim();
					CProducer newprd(Ai, q, NULL);
					ret.m_PS.insert(&newprd);
				}
			}
		}
		//去除Ai的直接左递归
		//将A的产生式分为两组，一组左递归，一组不是。
		CProducer_Set ps1 = ret.m_PS.GetDirectRcrsPrdcOfNonTerm(Ai);
		CProducer_Set ps2 = ret.m_PS.GetNonRcrsPrdcOfNonTerm(Ai);
		if (ps1.size() == 0)	//Ai的产生式没有左递归
		{
			continue;
		}
		if (ps2.size() == 0)
		{
			fprintf(stderr, "[%s][%d]文法推导无法结束!\n",
					__FILE__,
					__LINE__);
			exit(-1);
		}
		CNonTerminator AA = GetTmpNonTerminator();

		ret.m_PS.RmPrdcOfNonTerminator(Ai);
			
		ps1.begin_iterator();
		const CProducer * prd_ps1 = NULL;
		while ( (prd_ps1 = ps1.next() ) != NULL)
		{
			CSymbol_List a = prd_ps1->GetRight();
			a.removeAt(0);
			a.push_back(&AA);
			a.trim();
			CProducer newprd(AA, a, NULL);
			ret.m_PS.insert(&newprd);
		}

		ps2.begin_iterator();
		const CProducer * prd_ps2 = NULL;
		while ( (prd_ps2 = ps2.next() ) != NULL)
		{
			CSymbol_List b = prd_ps2->GetRight();
			b.push_back(&AA);
			b.trim();
			CProducer newprd(Ai, b, NULL);
			ret.m_PS.insert(&newprd);
		}
		CSymbol_List right;
		right.push_back(&CTerminator.EPSL);
		CProducer newprd(AA, right, NULL);
		ret.m_PS.insert(&newprd);
	}


	delete[] A;
	return ret;
}
/*
*得到一个临时的非终结符
*/
CNonTerminator CGrammar::GetTmpNonTerminator()
{
	static int index = 0;
	char buf[10];
	sprintf(buf, "TmpNT%d", index++);
	return CNonTerminator(buf);
}

CSymbol_List CGrammar::FIRST(const CSymbol_List &X, CProducer_Set G)
{
	CSymbol_List ret;
	CSymbol_List first;

	const CSymbol * psym = NULL;
	X.begin_iterator();
	while ( (psym = X.next()) != NULL)
	{
		first = CGrammar::FIRST(psym, G);
		bool bFirstContainEPSL = FALSE;
		const CSymbol * psymInFirst = NULL;
		first.begin_iterator();
		while ( (psymInFirst = first.next()) != NULL)
		{
			if (psymInFirst->GetSymType() == SYMBOL_TERMINATOR &&
				(*(CTerminator*)psymInFirst) == CTerminator::EPSL)
			{
				bFirstContainEPSL = TRUE;
				continue;
			}
			if (!ret.contain(psymInFirst))
			{
				ret.push_back(psymInFirst);
			}
		}
		if (!bFirstContainEPSL)
		{
			break;
		}
	}
	return ret;
}

int CGrammar::parse()
{
	StateOrSymbol_Stack	 stack;
	StateOrSymbol ele(0);
	stack.push(ele);
	CTerminator ip = lex();

	while (1)
	{
		stack.peek(ele);
		int s = ele.getState();	//a是栈顶的状态
		CTerminator a = ip;	//a是ip指向的符号

		ACTION action;
		action.state = s;
		action.terminator = a;
		if (m_table.GetAction(action) < 0)
		{
			//看能否按A->ε进行非常规处理
			action.terminator = CTerminator::EPSL;
			if (m_table.GetAction(action) < 0)//不能进行非常规处理
			{
				fprintf(stderr, "syntax error!lineno=[%d],yytext=[%s]\n",
					CLex::getlineno(),
					CLex::getyytext());
				return -1;
			}
			else
			{
				if (action.action[0] == 's')//移进
				{
					int dest_state = atoi(action.action + 1);
					StateOrSymbol  s2(dest_state);
					stack.push(s2);
					continue;
				}
				else
				{
					fprintf(stderr, "syntax error!lineno=[%d],yytext=[%s]\n",
							CLex::getlineno(),
							CLex::getyytext());
					return -1;
				}
			}
		}
		if (action.action[0] == 's')//移进
		{
			int dest_state = atoi(action.action + 1);
			StateOrSymbol s1(a), s2(dest_state);
			stack.push(s1);
			stack.push(s2);

			ip = lex();
		}
		else if (action.action[0] == 'r')//归约
		{
			int prd_number = atoi(action.action + 1);
			//得到A->β
			const CProducer* pprd = m_PS.GetProducerAt(prd_number);
			if (NULL == pprd)
			{
				fprintf(stderr, "[%s][%d]不可预见的错误!\n",
						__FILE__,
						__LINE__);
				return -1;
			}
			CProducer prd(*pprd);
			//弹出2 * |β|个元素
			int ele_num = prd.GetRight().size();
			const CSymbol * pfirstsymbol = prd.GetRight().GetSymbolAt(0);
			//如果是A->ε,不用弹出符号只弹出一个状态
			if (ele_num == 1 && 
				pfirstsymbol->GetSymType() == SYMBOL_TERMINATOR &&
				(*(CTerminator*)pfirstsymbol) == CTerminator::EPSL)
			{
				stack.pop();
			}
			else//否则弹出2 * |β|个元素
			{
				CSymbol_List right;
				for (int i = 0; i < ele_num; i++)
				{
					stack.pop();//弹状态

					StateOrSymbol ele_in_stack(0);
					stack.peek(ele_in_stack);
					right.push_front(ele_in_stack.getSymbol());
					stack.pop();//弹符号
				}
				//右部中个符号的属性都存在在栈中
				prd.SetRight(right);
			}
			//执行产生式的翻译动作,可能修改prd左部的属性
			PRODUCER_FUNC func = prd.GetFunc();
			if (func != NULL)
			{
				func(&prd);
			}
			//ss是栈顶的状态
			StateOrSymbol ss(0);
			stack.peek(ss);
			//把A和goto[ss, A]压栈
			StateOrSymbol s1(prd.GetLeft());
			GOTO gt;
			gt.state = ss.getState();
			gt.nonterminator = prd.GetLeft();
			if (m_table.GetGoto(gt) < 0)
			{
				fprintf(stderr, "syntax error!lineno=[%d],yytext=[%s]\n",
					CLex::getlineno(),
					CLex::getyytext());
				return -1;
			}
			StateOrSymbol s2(gt.gotostate);
			stack.push(s1);
			stack.push(s2);

			printf("归约>>>> %s\n", prd.ToString().c_str()); 

		
			
		}
		else if (strcmp(action.action, "acc") == 0)//接受
		{
			return 0;
		}
		else
		{
			fprintf(stderr, "分析表中有不可识别的动作!\n");
			return -1;
		}
	}

	return 0;
}


CTerminator CGrammar::lex() const
{
	int token = CLex::lex();
	switch (token)
	{
		case IDX_ID : return AllToken::ID;
		case IDX_CONST_STRING: return AllToken::CONST_STRING;
		case IDX_CONST_INTEGER: return AllToken::CONST_INTEGER;
		case IDX_CONST_FLOAT: return AllToken::CONST_FLOAT;
		case IDX_FUNCTION: return AllToken::FUNCTION;
		case IDX_IF: return AllToken::IF;
		case IDX_THEN: return AllToken::THEN;
		case IDX_ELSE: return AllToken::ELSE;
		case IDX_ENDIF: return AllToken::ENDIF;
		case IDX_WHILE: return AllToken::WHILE;
		case IDX_DO: return AllToken::DO;
		case IDX_ENDWHILE: return AllToken::ENDWHILE;
		case IDX_INTEGER: return AllToken::INTEGER;
		case IDX_STRING: return AllToken::STRING;
		case IDX_FLOAT: return AllToken::FLOAT;
		case IDX_RETURN: return AllToken::RETURN;
		case IDX_BEGIN_FLOW: return AllToken::BEGIN_FLOW;
		case IDX_END_FLOW: return AllToken::END_FLOW;
		case IDX_RUN: return AllToken::RUN;
		case IDX_FOR: return AllToken::FOR;
		case IDX_ENDFOR: return AllToken::ENDFOR;
		case IDX_CONTINUE: return AllToken::CONTINUE;
		case IDX_BREAK: return AllToken::BREAK;
		case IDX_REPEAT: return AllToken::REPEAT;
		case IDX_UNTIL: return AllToken::UNTIL;
		case IDX_SWITCH: return AllToken::SWITCH;
		case IDX_ENDSWITCH: return AllToken::ENDSWITCH;
		case IDX_CASE: return AllToken::CASE;
		case IDX_OR: return AllToken::OR;
		case IDX_AND: return AllToken::AND;
		case IDX_LT: return AllToken::LT;
		case IDX_LE: return AllToken::LE;
		case IDX_EQ: return AllToken::EQ;
		case IDX_NE: return AllToken::NE;
		case IDX_GT: return AllToken::GT;
		case IDX_GE: return AllToken::GE;
		case IDX_UMINUS: return AllToken::UMINUS;
		case IDX_NOT: return AllToken::NOT;
		case IDX_MEMBLOCK: return AllToken::MEMBLOCK;
		case IDX_SEMI: return AllToken::SEMI;
		case IDX_L_SQ_BRACKET: return AllToken::L_SQ_BRACKET;
		case IDX_R_SQ_BRACKET: return AllToken::R_SQ_BRACKET;
		case IDX_ASSIGN: return AllToken::ASSIGN;
		case IDX_ADD: return AllToken::ADD;
		case IDX_SUB: return AllToken::SUB;
		case IDX_MUL: return AllToken::MUL;
		case IDX_DIV: return AllToken::DIV;
		case IDX_MOD: return AllToken::MOD;
		case IDX_L_BRACKET: return AllToken::L_BRACKET;
		case IDX_R_BRACKET: return AllToken::R_BRACKET;
		case IDX_COMMA: return AllToken::COMMA;
		case IDX_COLON: return AllToken::COLON;
		case 0	: return CTerminator::FINIS;
		default: break;
	}
	fprintf(stderr, "[%s][%d]不可识别的记号!\n",
			__FILE__,
			__LINE__);
	exit(-1);
}

void CGrammar::CalculateAnalyseTable()
{
	CGrammar GG = this->ClearRecursion();
	//计算初始状态
	CItem_Set I0;
	{ 
		CProducer_Set ps = m_PS.GetProducerOfNonTerm(m_StartSymbol);
		if (ps.size() != 1)
		{
			fprintf(stderr, "[%s][%d]开始符号的产生式的个数不为1，请检查是否已经扩展了文法!\n",
					__FILE__,
					__LINE__);
			exit(-1);
		}
		ps.begin_iterator();
		const CProducer * pprd =  NULL;
		pprd = ps.next();
		CSymbol_List right = pprd->GetRight();
		if (right.size() != 1)
		{
			fprintf(stderr, "[%s][%d]开始符号的产生式不为正确，请检查是否已经扩展了文法!\n",
					__FILE__,
					__LINE__);
			exit(-1);			
		}
		right.push_front(&CDot());
		CProducer newprd(pprd->GetLeft(), right, NULL);
		CItem item(newprd, CTerminator::FINIS);
		I0.insert(&item);
		I0 = Closure(I0, m_PS, GG.m_PS);
#ifdef _DEBUG
		{
			I0.begin_iterator();
			const CItem* ppp = NULL;
			while ( (ppp = I0.next()) != NULL)
			{
				printf("-----%s\n", ppp->ToString().c_str());
			}
		}
#endif
	}
	
	CalculateAnalyseTable(m_table, m_SL, m_PS, I0, m_StartSymbol, GG.m_PS);
}

int CGrammar::WriteAnalyseTableToFile(const char *filename)
{
	return m_table.WriteToFile(filename);
}

int CGrammar::ReadAnalyseTableFrmFile(const char *filename)
{
	return m_table.ReadFrmFile(filename);
}
