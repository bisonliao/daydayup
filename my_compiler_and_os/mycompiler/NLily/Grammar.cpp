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

	//���X->����һ������ʽ�����ż���FIRST(X)
	CSymbol_List right;
	right.push_back(&CTerminator::EPSL);
	CProducer prd(*(CNonTerminator*)X, right, NULL);
	if (G.contain(prd))//���prd��һ������ʽ
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
		// a) FIRST(Y1)�����з��Ŷ���FIRST(X)��
		right = pprd->GetRight();
		right.begin_iterator();
		const CSymbol* Y1 = right.next();

		///Ȩ�˼�������ж�
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
	
		// b)������ĳ��i, a����FIRST(Yi)�Ҧ�����
		//FIRST(Y1),...,FIRST(Yi-1),��a����FIRST(X)
		right = pprd->GetRight();
		right.begin_iterator();
		const CSymbol * Yi = NULL;
		bool bAllHasEPSL = TRUE;//right��ÿ�����ŵ�first�������Ц�
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
			break;//��Ҳ����Ҫ��������
		}
		// c)��������е�j=1,2,...,k, ����FIRST(Yk)�У�
		//��ô���żӵ�FIRST(X)��
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
*����Ŀ��I�ıհ���G���ķ��Ĳ���ʽ�ļ���, GG�����ķ��ȼ۵ķǵݹ��ķ��Ĳ���ʽ�ļ���
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
		while ( (item = J.next()) != NULL)//ÿ��[A->��.B��,a]��Ŀ
		{
			int dotIndex = item->GetProducer().GetDotIndex();
			const CSymbol * B = item->GetProducer().GetSymbolAt(dotIndex + 1);
			if (NULL == B)
			{
				continue;
			}
			//��FIRST(��a)
			CSymbol_List first_Beta_a;
			CSymbol_List beta;
			//�ȵõ�beta
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
				if ( !(pprd->GetLeft() == (*(CNonTerminator*)B) ) )//����ʽ�󲿲�ΪB
				{
					continue;
				}
				//����[B->.��, b]��Ŀ
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
		while ( (item = J.next()) != NULL)//ÿ����Ŀ[A->��.B��,a]
		{
			int dotIndex = item->GetProducer().GetDotIndex();
			const CSymbol * B = item->GetProducer().GetSymbolAt(dotIndex + 1);
			if (NULL == B)
			{
				continue;
			}
			//��FIRST(��a)
			CSymbol_List first_Beta_a;
			CSymbol_List beta;
			//�ȵõ�beta
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
			while ( (pprd = G.next()) != NULL)//ÿ������ʽ
			{
				if ( !(pprd->GetLeft() == (*(CNonTerminator*)B) ) )//����ʽ�󲿲�ΪB
				{
					continue;
				}
				//����[B->.��, b]��Ŀ
				CSymbol_List garma = pprd->GetRight();
				garma.push_front(&(CDot()));

				first_Beta_a.begin_iterator();
				const CSymbol * b = NULL;
				while ( (b = first_Beta_a.next()) != NULL)//ÿ���ս����
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
		//������pXȷʵ�͵���X
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
*	C:��Ŀ���ļ���
*	I0:��ʼ��Ŀ��
*	SL:�ķ��������ķ����ŵļ���
*	G:�ķ������в���ʽ�ļ���
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
		while ( ( I = C.next()) != NULL)//ÿ����Ŀ��
		{
			SL.begin_iterator();
			const CSymbol* X = NULL;
			while ( (X = SL.next()) !=  NULL)//ÿ���ķ�����
			{
				CItem_Set newitemset = CGrammar::Goto(*I, X, G, GG);
				if (newitemset.size() == 0)//Ϊ��
				{
					continue;
				}
				if ( C.contain(newitemset) )//�Ѿ�����C��
				{
					continue;
				}
				C.add(newitemset);
				bNewEleAdded = TRUE;
				printf("----------����һ����Ŀ��%d\n", ++count);
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

	
		while ( ( I = jjj.next()) != NULL)//ÿ����Ŀ��
		{
	
			SL.begin_iterator();
			const CSymbol* X = NULL;
			while ( (X = SL.next()) !=  NULL)//ÿ���ķ�����
			{
				CItem_Set newitemset = CGrammar::Goto(*I, X, G, GG);
				if (newitemset.size() == 0)//Ϊ��
				{
					continue;
				}
				if ( C.contain(newitemset) )//�Ѿ�����C��
				{
					continue;
				}
				C.add(newitemset);
				ttt.add(newitemset);
				bNewEleAdded = TRUE;
				printf("----------����һ����Ŀ��%d\n", ++count);
			}
		}
	}
	while (bNewEleAdded);
}

/*
*table:������
*
*/

void CGrammar::CalculateAnalyseTable(CAnalyseTable& table, const CSymbol_List SL, const CProducer_Set G, const CItem_Set I0, const CNonTerminator  StartSymbol, const CProducer_Set& GG)
{
	table.clear();
	//1.������Ŀ���淶��
	items C;
	printf("----------��ʼ������Ŀ���淶��\n");
	CGrammar::Items(C, I0, SL, G, GG);
	printf("----------������Ŀ���淶�����\n");


	//2.״̬�Ķ���
	printf("----------��ʼ����ACTION\n");
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
						fprintf(stderr, "[%s][%d]�ķ�����LR(1)�ķ�,����������ͻ!\n",
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
						fprintf(stderr, "[%s][%d]�ķ�����LR(1)�ķ�,����������ͻ!\n",
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
					fprintf(stderr, "[%s][%d]����ʽ��Ч!\n",
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
						fprintf(stderr, "[%s][%d]�ķ�����LR(1)�ķ�,����������ͻ!\n",
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
	printf("----------����ACTION����\n");
	
	//3.״̬��ת��
	printf("----------��ʼ����GOTO\n");
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
					fprintf(stderr, "[%s][%d]�ķ�����LR(1)�ķ�!\n",
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
	printf("----------����GOTO����\n");
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
*��������ķ��ȼ۵�����ݹ��ķ�
*/
CGrammar CGrammar::ClearRecursion() const
{
	CGrammar ret = *this;

	///////////////////////////////////
	//�㷨4.1 ������ӵݹ�
	//1.��ĳ��˳�����з��ս����A1,A2,...,An
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
			fprintf(stderr, "[%s][%d]�ڴ����ʧ��!\n",
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
			
			CProducer_Set Ai_prdset = ret.m_PS.RmAiAjProducer(Ai, Aj);//ɾ������Ai->Aj �õĲ���ʽ

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
		//ȥ��Ai��ֱ����ݹ�
		//��A�Ĳ���ʽ��Ϊ���飬һ����ݹ飬һ�鲻�ǡ�
		CProducer_Set ps1 = ret.m_PS.GetDirectRcrsPrdcOfNonTerm(Ai);
		CProducer_Set ps2 = ret.m_PS.GetNonRcrsPrdcOfNonTerm(Ai);
		if (ps1.size() == 0)	//Ai�Ĳ���ʽû����ݹ�
		{
			continue;
		}
		if (ps2.size() == 0)
		{
			fprintf(stderr, "[%s][%d]�ķ��Ƶ��޷�����!\n",
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
*�õ�һ����ʱ�ķ��ս��
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
		int s = ele.getState();	//a��ջ����״̬
		CTerminator a = ip;	//a��ipָ��ķ���

		ACTION action;
		action.state = s;
		action.terminator = a;
		if (m_table.GetAction(action) < 0)
		{
			//���ܷ�A->�Ž��зǳ��洦��
			action.terminator = CTerminator::EPSL;
			if (m_table.GetAction(action) < 0)//���ܽ��зǳ��洦��
			{
				fprintf(stderr, "syntax error!lineno=[%d],yytext=[%s]\n",
					CLex::getlineno(),
					CLex::getyytext());
				return -1;
			}
			else
			{
				if (action.action[0] == 's')//�ƽ�
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
		if (action.action[0] == 's')//�ƽ�
		{
			int dest_state = atoi(action.action + 1);
			StateOrSymbol s1(a), s2(dest_state);
			stack.push(s1);
			stack.push(s2);

			ip = lex();
		}
		else if (action.action[0] == 'r')//��Լ
		{
			int prd_number = atoi(action.action + 1);
			//�õ�A->��
			const CProducer* pprd = m_PS.GetProducerAt(prd_number);
			if (NULL == pprd)
			{
				fprintf(stderr, "[%s][%d]����Ԥ���Ĵ���!\n",
						__FILE__,
						__LINE__);
				return -1;
			}
			CProducer prd(*pprd);
			//����2 * |��|��Ԫ��
			int ele_num = prd.GetRight().size();
			const CSymbol * pfirstsymbol = prd.GetRight().GetSymbolAt(0);
			//�����A->��,���õ�������ֻ����һ��״̬
			if (ele_num == 1 && 
				pfirstsymbol->GetSymType() == SYMBOL_TERMINATOR &&
				(*(CTerminator*)pfirstsymbol) == CTerminator::EPSL)
			{
				stack.pop();
			}
			else//���򵯳�2 * |��|��Ԫ��
			{
				CSymbol_List right;
				for (int i = 0; i < ele_num; i++)
				{
					stack.pop();//��״̬

					StateOrSymbol ele_in_stack(0);
					stack.peek(ele_in_stack);
					right.push_front(ele_in_stack.getSymbol());
					stack.pop();//������
				}
				//�Ҳ��и����ŵ����Զ�������ջ��
				prd.SetRight(right);
			}
			//ִ�в���ʽ�ķ��붯��,�����޸�prd�󲿵�����
			PRODUCER_FUNC func = prd.GetFunc();
			if (func != NULL)
			{
				func(&prd);
			}
			//ss��ջ����״̬
			StateOrSymbol ss(0);
			stack.peek(ss);
			//��A��goto[ss, A]ѹջ
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

			printf("��Լ>>>> %s\n", prd.ToString().c_str()); 

		
			
		}
		else if (strcmp(action.action, "acc") == 0)//����
		{
			return 0;
		}
		else
		{
			fprintf(stderr, "���������в���ʶ��Ķ���!\n");
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
	fprintf(stderr, "[%s][%d]����ʶ��ļǺ�!\n",
			__FILE__,
			__LINE__);
	exit(-1);
}

void CGrammar::CalculateAnalyseTable()
{
	CGrammar GG = this->ClearRecursion();
	//�����ʼ״̬
	CItem_Set I0;
	{ 
		CProducer_Set ps = m_PS.GetProducerOfNonTerm(m_StartSymbol);
		if (ps.size() != 1)
		{
			fprintf(stderr, "[%s][%d]��ʼ���ŵĲ���ʽ�ĸ�����Ϊ1�������Ƿ��Ѿ���չ���ķ�!\n",
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
			fprintf(stderr, "[%s][%d]��ʼ���ŵĲ���ʽ��Ϊ��ȷ�������Ƿ��Ѿ���չ���ķ�!\n",
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
