#include "script.h"

#include <stack>
#include "var.h"
#include "mem.h"


int RunScript(const string & scriptname, const list<CVar> &args, CVar & rtnval);

CScript::CScript()
{
}
void CScript::Clear()
{
	m_vInstructs.clear();
	m_mLabels.clear();
}
CScript::CScript(const CScript & another)
{
	m_vInstructs = another.m_vInstructs;
	m_mLabels = another.m_mLabels;
	m_mem = another.m_mem;
}
const CScript & CScript::operator=(const CScript & another)
{
	m_vInstructs = another.m_vInstructs;
	m_mLabels = another.m_mLabels;
	m_mem = another.m_mem;
}
int CScript::PushInstruct(const vector<string> & instructs)
{
	int size = instructs.size();
	if (strncmp("!!!BEGIN", instructs[0].c_str(), 8) != 0 ||
		strncmp("!!!END", instructs[size-1].c_str(), 6) !=0 )
	{
		return -1;
	}
	for (int i = 0; i < size; ++i)
	{
		string instr = instructs[i];
		if (instr.length() <= 0)
		{
			return -1;
		}
		m_vInstructs.push_back(instr);
	
	
		if (instr.length() >= 7 &&
			instr.substr(0, 5) == string("LABEL") )
		{
			// LABEL l123
			int nLabelIndex = atoi(instr.substr(7).c_str());
			int nPos = m_vInstructs.size() - 1;
	
			m_mLabels.insert( std::pair<int, int>(nLabelIndex, nPos) );
		}
	}
	return 0;
}
int CScript::Run(const list<CVar> &args, CVar & rtnval)
{
	//���öνű��Ĳ�����ʼ�����ڴ���
	list<CVar>::const_iterator args_it;
	int args_idx;
	for (args_idx = 1, args_it = args.begin(); args_it != args.end(); 
		++args_it, ++args_idx)
	{
		char argname[100];
		sprintf(argname, "$%d", args_idx);
		string aaa = argname;
		CVar vvv = *args_it;
		m_mem.SetVar(aaa, vvv);
	}

	//��ʼ����
	stack<CVar> runstk; //����ջ	
	int i = 0;
	for (unsigned int index = 0; index < m_vInstructs.size(); ++index, ++i)
	{
		string instruct = m_vInstructs.at(index);
		string action;
		string object; 
		//�ֽ�������Ͷ���
		int pos = instruct.find(' ');
		if (pos == -1)
		{
			action = instruct;
			object = "";
		}
		else
		{
			action = instruct.substr(0, pos);
			object = instruct.substr(pos+1);
		}

		if (action == "!!!END")
		{
			break;
		}
		else if (action == "!!!BEGIN")
		{
		}
		else if (action == "LABEL")
		{
		}
		else if (action == "PUSH")
		{
			string substr0 = object.substr(0, 1);
			CVar var;
			if (substr0 == "^") //��������
			{
				var.m_nType = CVar::T_INT;
				var.m_intval = atoi(object.substr(1).c_str());
				runstk.push(var);
			}
			else if (substr0 == "v" ||
				substr0 == "$") //�û�����������ʱ����
			{
				m_mem.GetVar(object, var);
				runstk.push(var);
			}
			else if (substr0 == "#") //�ַ�������
			{
				var.m_nType = CVar::T_STR;
				var.m_stringval = object.substr(2, object.length()-3);
				runstk.push(var);
			}
			else if (substr0 == "%") //���㳣��
			{
				var.m_nType = CVar::T_FLOAT;
				var.m_floatval = atof(object.substr(1).c_str());
				runstk.push(var);
			}
			else 
			{
				fprintf(stderr, "ָ��Ƿ�! %s\n", instruct.c_str());
				return -1;
			}
		}
		else if (action == "PUSARY") //����Ԫ��ѹջ
		{
			if ( object.substr(0, 1) == "$") //�û�����������ʱ����
			{
				CVar var;
				CVar offset = runstk.top();
				runstk.pop();
				m_mem.GetVar(object, var, offset.m_intval);
				runstk.push(var);
			}
			else 
			{
				fprintf(stderr, "ָ��Ƿ�! %s\n", instruct.c_str());
				return -1;
			}
		}
		else if (action == "SAV")
		{
			if (object.substr(0,1) == "$" ||
				object.substr(0,1) == "v")
			{
				if (runstk.empty())
				{
					fprintf(stderr, "�������ʱ����ջΪ��! %s\n",
						instruct.c_str());
					return -1;
				}
				CVar var = runstk.top();
				runstk.pop();
				m_mem.SetVar(object, var);
			}
			else
			{
				fprintf(stderr, "�Ƿ�ָ��! %s\n", instruct.c_str());
				return -1;
			}
		}
		else if (action == "SAVARY") /*��������Ԫ��*/
		{
			if (object.substr(0,1) == "$" )
			{
				if (runstk.size() < 2)
				{
					fprintf(stderr, "��������Ԫ��ʱ����ջ�ڲ�������������! %s\n",
						instruct.c_str());
					return -1;
				}
				CVar offset = runstk.top();
				runstk.pop();
				CVar var = runstk.top();
				runstk.pop();
				m_mem.SetVar(object, var, offset.m_intval);
			}
			else
			{
				fprintf(stderr, "�Ƿ�ָ��! %s\n", instruct.c_str());
				return -1;
			}
		}
		else if (action == "CLEAR")
		{
			while (!runstk.empty())
			{
				runstk.pop();
			}
		}
		else if (action == "CALL")
		{
			list<CVar> arglist;
			if (runstk.empty())
			{
				fprintf(stderr, "��������ʱ����ջΪ��! %s\n",
					instruct.c_str());
				return -1;
			}
			CVar var;
			var = runstk.top();
			runstk.pop();
			if (var.m_nType != CVar::T_INT ||
				var.m_intval < 0)
			{
				fprintf(stderr, "��������ʱ�����������Ϸ�! %s\n",
					instruct.c_str());
				return -1;
			}
			for (int i = 0; i < var.m_intval; ++i)
			{
				if (runstk.empty())
				{
					fprintf(stderr, "��������ʱ����ջΪ��! %s\n",
						instruct.c_str());
					return -1;
				}
				CVar argvar = runstk.top();
				runstk.pop();

				arglist.push_front(argvar);
			}
			var = RunFunction(object, arglist);
			runstk.push(var);

		}
		else if (action == "GOTO")
		{
			int LabIdx = atoi(object.substr(1).c_str());
			map<int, int>::const_iterator labit = m_mLabels.find(LabIdx);
			if (labit == m_mLabels.end())
			{
				fprintf(stderr, "�Ƿ���ת���! %s\n", instruct.c_str());
				return -1;
			}
			int LabPos = labit->second;
			index = LabPos;
		}
		else if (action == "GOTOFALSE")
		{
			if (runstk.empty())
			{
				fprintf(stderr, "ִ����ת���ʱ����ջΪ��! %s\n", instruct.c_str());
				return -1;
			}
			CVar var = runstk.top();
			runstk.pop();
			int nCondition = 0;
			if (var.m_nType == CVar::T_INT)
			{
				nCondition = var.m_intval;
			}
			else if (var.m_nType == CVar::T_FLOAT)
			{
				nCondition = var.m_floatval;
			}
			else 
			{
				nCondition = (var.m_stringval.length() > 0 ? 1:0);
			}
			if (nCondition == 0)
			{
				int LabIdx = atoi(object.substr(1).c_str());
				map<int, int>::const_iterator labit = m_mLabels.find(LabIdx);
				if (labit == m_mLabels.end())
				{
					fprintf(stderr, "�Ƿ���ת���! %s\n", instruct.c_str());
					return -1;
				}
				int LabPos = labit->second;
				index = LabPos;
			}
		}
		else if (action == "ADD")
		{
			if (runstk.size() < 2)
			{
				fprintf(stderr, "������̫��! %s\n", instruct.c_str());
				return -1;
			}
			CVar lop, rop;
			rop = runstk.top();
			runstk.pop();
			lop = runstk.top();
			runstk.pop();

			runstk.push( lop + rop );
		}
		else if (action == "SUB")
		{
			if (runstk.size() < 2)
			{
				fprintf(stderr, "������̫��! %s\n", instruct.c_str());
				return -1;
			}
			CVar lop, rop;
			rop = runstk.top();
			runstk.pop();
			lop = runstk.top();
			runstk.pop();

			runstk.push( lop - rop );
		}
		else if (action == "MUL")
		{
			if (runstk.size() < 2)
			{
				fprintf(stderr, "������̫��! %s\n", instruct.c_str());
				return -1;
			}
			CVar lop, rop;
			rop = runstk.top();
			runstk.pop();
			lop = runstk.top();
			runstk.pop();

			runstk.push( lop * rop );
		}
		else if (action == "DIV")
		{
			if (runstk.size() < 2)
			{
				fprintf(stderr, "������̫��! %s\n", instruct.c_str());
				return -1;
			}
			CVar lop, rop;
			rop = runstk.top();
			runstk.pop();
			lop = runstk.top();
			runstk.pop();

			runstk.push( lop / rop );
		}
		else if (action == "LT")
		{
			if (runstk.size() < 2)
			{
				fprintf(stderr, "������̫��! %s\n", instruct.c_str());
				return -1;
			}
			CVar lop, rop;
			rop = runstk.top();
			runstk.pop();
			lop = runstk.top();
			runstk.pop();

			runstk.push( lop < rop );
		}
		else if (action == "GT")
		{
			if (runstk.size() < 2)
			{
				fprintf(stderr, "������̫��! %s\n", instruct.c_str());
				return -1;
			}
			CVar lop, rop;
			rop = runstk.top();
			runstk.pop();
			lop = runstk.top();
			runstk.pop();

			runstk.push( lop > rop );
		}
		else if (action == "GE")
		{
			if (runstk.size() < 2)
			{
				fprintf(stderr, "������̫��! %s\n", instruct.c_str());
				return -1;
			}
			CVar lop, rop;
			rop = runstk.top();
			runstk.pop();
			lop = runstk.top();
			runstk.pop();

			runstk.push( lop >= rop );
		}
		else if (action == "LE")
		{
			if (runstk.size() < 2)
			{
				fprintf(stderr, "������̫��! %s\n", instruct.c_str());
				return -1;
			}
			CVar lop, rop;
			rop = runstk.top();
			runstk.pop();
			lop = runstk.top();
			runstk.pop();

			runstk.push( lop <= rop );
		}
		else if (action == "EQ")
		{
			if (runstk.size() < 2)
			{
				fprintf(stderr, "������̫��! %s\n", instruct.c_str());
				return -1;
			}
			CVar lop, rop;
			rop = runstk.top();
			runstk.pop();
			lop = runstk.top();
			runstk.pop();

			runstk.push( lop == rop );
		}
		else if (action == "NE")
		{
			if (runstk.size() < 2)
			{
				fprintf(stderr, "������̫��! %s\n", instruct.c_str());
				return -1;
			}
			CVar lop, rop;
			rop = runstk.top();
			runstk.pop();
			lop = runstk.top();
			runstk.pop();

			runstk.push( lop != rop );
		}
		else if (action == "NOT")
		{
			if (runstk.empty())
			{
				fprintf(stderr, "������̫��! %s\n", instruct.c_str());
				return -1;
			}
			CVar op;
			op = runstk.top();
			runstk.pop();

			runstk.push( !op );
		}
		else
		{
			fprintf(stderr, "�Ƿ�ָ��! %s\n", instruct.c_str());
			return -1;
		}
	}
	m_mem.GetVar("$OUTDATA", rtnval);
	return 0;
}
//�������õĺ���
CVar CScript::RunFunction(const string & funcname, const list<CVar> &arglist)
{
	if (funcname == "print") //�������
	{
		list<CVar>::const_iterator it ;
		for (it = arglist.begin(); it != arglist.end(); ++it)
		{
			printf("%s", (*it).ToString().c_str());
		}
	}
	else if (funcname == "int")  //��һ������ת��Ϊ������
	{
		if (arglist.empty())
		{
			return CVar();
		}
		list<CVar>::const_iterator it = arglist.begin();
		CVar vvv;
		vvv.m_nType = CVar::T_INT;
		if ( it->m_nType == CVar::T_STR)
		{
			vvv.m_intval = atoi((*it).m_stringval.c_str());
		}
		else if ( it->m_nType == CVar::T_INT)
		{
			vvv.m_intval = it->m_intval;
		}
		else if ( it->m_nType == CVar::T_FLOAT)
		{
			vvv.m_intval = it->m_floatval;
		}
		return vvv;
	}
	else if (funcname == "float") //������ת��Ϊ������
	{
		CVar vvv;
		vvv.m_nType = CVar::T_FLOAT;
		if (arglist.empty())
		{
			vvv.m_floatval = 0.0;
			return vvv;
		}
		list<CVar>::const_iterator it = arglist.begin();
		if ( it->m_nType == CVar::T_STR)
		{
			vvv.m_floatval = atof((*it).m_stringval.c_str());
		}
		else if ( it->m_nType == CVar::T_INT)
		{
			vvv.m_floatval = it->m_intval;
		}
		else if ( it->m_nType == CVar::T_FLOAT)
		{
			vvv.m_floatval = it->m_floatval;
		}
		return vvv;
	}
	else if (funcname == "str")  //������ת��Ϊ�ַ�����
	{
		CVar vvv;
		vvv.m_nType = CVar::T_STR;
		if (arglist.empty())
		{
			vvv.m_stringval = "";
			return vvv;
		}
		list<CVar>::const_iterator it = arglist.begin();
		if ( it->m_nType == CVar::T_STR)
		{
			vvv.m_stringval = it->m_stringval;
		}
		else if ( it->m_nType == CVar::T_INT)
		{
			char tmpbuf[100];
			sprintf(tmpbuf, "%d", it->m_intval);
			vvv.m_stringval = tmpbuf;
		}
		else if ( it->m_nType == CVar::T_FLOAT)
		{
			char tmpbuf[100];
			sprintf(tmpbuf, "%f", it->m_floatval);
			vvv.m_stringval = tmpbuf;
		}
		return vvv;
	}
	else  //��������һ������
	{
		CVar vvv;
		if (RunScript(funcname, arglist, vvv) == 0)
		{
			return vvv;
		}
	}
	return CVar();
}

