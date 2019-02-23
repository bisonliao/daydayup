#include "Variable_Stack.h"

Variable_Stack::Variable_Stack() 
{
	this->m_nBufSize = 100;
	this->m_nEleNum = 0;
	this->m_pHdr = new Variable[100];
	if (this->m_pHdr == NULL)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
				__FILE__,
				__LINE__);
		exit(-1);
	}
}
Variable_Stack::Variable_Stack(Variable_Stack &ss) 
{
	this->m_nEleNum = ss.m_nEleNum;
	this->m_nBufSize = ss.m_nBufSize;
	this->m_pHdr = new Variable[m_nBufSize];
	if (this->m_pHdr == NULL)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
				__FILE__,
				__LINE__);
		exit(-1);
	}
	for (int i = 0; i < m_nEleNum; i++)
	{
		this->m_pHdr[i] = ss.m_pHdr[i];
	}
}
Variable_Stack::~Variable_Stack()
{
	if (NULL != m_pHdr)
	{
		delete[] m_pHdr;
	}
}
Variable_Stack&Variable_Stack::operator=(const Variable_Stack&ss) 
{
	if (this == &ss)
	{
		return *this;
	}
	if (m_pHdr != NULL)
	{
		delete[] m_pHdr;
		m_pHdr = NULL;
	}
	this->m_nEleNum = ss.m_nEleNum;
	this->m_nBufSize = ss.m_nBufSize;
	this->m_pHdr = new Variable[m_nBufSize];
	if (this->m_pHdr == NULL)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
				__FILE__,
				__LINE__);
		exit(-1);
	}
	for (int i = 0; i < m_nEleNum; i++)
	{
		this->m_pHdr[i] = ss.m_pHdr[i];
	}
	return *this;
}
bool Variable_Stack::push(const Variable & ele) 
{
	if (isFull())
	{
		if (enlarge() != 0)
		{
			return FALSE;
		}
	}
	m_pHdr[m_nEleNum++] = ele;
	return TRUE;
}
bool Variable_Stack::pop(Variable& ele)
{
	if (m_nEleNum <= 0)
	{
		return FALSE;
	}
	ele = m_pHdr[--m_nEleNum];
	/*
	* 如果变量类型为TYPE_STRING或者TYPE_MEMBLOCK，可能有大量的堆空间没有释放
	* 所以应该主动进行清除 
	*/
	m_pHdr[m_nEleNum].clear();



	return TRUE;		
}
bool Variable_Stack::peek(Variable& ele)
{
	if (m_nEleNum <= 0)
	{
		return FALSE;
	}
	ele = m_pHdr[m_nEleNum - 1];
	return TRUE;	
}
bool Variable_Stack::isEmpty()
{
	return (m_nEleNum <= 0);
}
int Variable_Stack::getSize()
{
	return m_nEleNum;
}
bool Variable_Stack::isFull()
{
	return (m_nEleNum >= m_nBufSize);
}
int Variable_Stack::enlarge()
{
	Variable * ptr = new Variable[m_nBufSize + 100];
	if (NULL == ptr)
	{
		return -1;
	}
	for (int i = 0; i < m_nEleNum; i++)
	{
		ptr[i] = m_pHdr[i];
	}
		
	Variable * tmp = m_pHdr;
	delete[] tmp;
	m_pHdr = ptr;
	m_nBufSize += 100;
	return 0;
}
/*从栈顶往栈底查看depth个变量，如果某个变量的名字等于name，那么返回成功*/
bool Variable_Stack::FindVarByNameFrmTop(const AnsiString& name, 
	int depth, 
	Variable& ele)
{
	if (depth < 0 || depth > m_nEleNum)
	{
		return FALSE;
	}
	for (int i = 0; i < depth; i++)
	{
		Variable v = m_pHdr[m_nEleNum - 1 - i];
		if (v.getName() == name)
		{
			ele = v;
			return TRUE;
		}
	}
	return FALSE;
}
bool Variable_Stack::ModifyVarByNameFrmTop(const AnsiString& name, 
	int depth, 
	const Variable& ele)
{
	if (depth < 0 || depth > m_nEleNum)
	{
		return FALSE;
	}
	#ifdef _DEBUG
	/*
		if (name == "mb2")
		{
			printf("ModifyVarByNameFrmTop>>>>[%s][%d]:%s\n", name.c_str(), depth, ele.toString().c_str());
		}
		*/
	#endif
	for (int i = 0; i < depth; i++)
	{
		Variable v = m_pHdr[m_nEleNum - 1 - i];
		if (v.getName() == name)
		{
			int  index = m_nEleNum - 1 - i;
			if (m_pHdr[index].getType() == ele.getType())
			{
				m_pHdr[index] = ele;
				m_pHdr[index].setName(name);
			}
			else if (m_pHdr[index].getType() == TYPE_INTEGER &&
					ele.getType() == TYPE_FLOAT)
			{
				m_pHdr[index].setInteger((int)(ele.getFloat()));
			}
			else if (m_pHdr[index].getType() == TYPE_FLOAT &&
					ele.getType() == TYPE_INTEGER)
			{
				m_pHdr[index].setFloat((float)(ele.getInteger()));
			}
			return TRUE;
		}
	}
	return FALSE;
}
void Variable_Stack::removeAll()
{
	m_nEleNum = 0;
}
