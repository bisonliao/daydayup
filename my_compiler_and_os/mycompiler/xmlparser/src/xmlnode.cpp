#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "xmlnode.h"

XmlNode::XmlNode()
{

}

XmlNode::~XmlNode()
{

}
const AnsiString & XmlNode::GetNodeName() const
{
	return m_NodeName;
}
const AnsiString & XmlNode::GetNodeProperty() const
{
	return m_NodeProperty;
}
const AnsiString & XmlNode::GetNodeValue() const
{
	return m_NodeValue;
}
const list<XmlNode*> & XmlNode::GetNodeChildList() const
{
	return m_ChildList;
}
void  XmlNode::SetNodeName(const AnsiString & name)
{
	m_NodeName = name;
}
void  XmlNode::SetNodeValue(const AnsiString & value)
{
	m_NodeValue = value;
}
void  XmlNode::SetNodeProperty(const AnsiString & property)
{
	m_NodeProperty = property;
}
int  XmlNode::AddChild(XmlNode * pNode)
{
	if (pNode == NULL)
	{
		return -1;
	}
	m_ChildList.push_back(pNode);
	pNode->SetParent(this);
	return 0;
}
XmlNode * XmlNode::GetParent()  const
{
	return m_Parent;
}
void  XmlNode::SetParent(XmlNode* parent)
{
	m_Parent = parent;
}
/*
*取第index个名字为nodename的子节点的指针, index从0开始
*失败返回NULL
*/
XmlNode * XmlNode::GetChildByNodeName(const char * nodename, int index) const
{
	int i = -1;
	if (NULL == nodename)
	{
		return NULL;
	}
	list<XmlNode*>::const_iterator it;
	for ( it = m_ChildList.begin(); it != m_ChildList.end(); ++it)
	{
		XmlNode * ptr = *it;
		if (ptr->GetNodeName() == nodename)
		{
			++i;
			if (i == index)
			{
				return ptr;
			}
		}
	}
	return NULL;
}
/*
* 确保第index个名字为nodename的子节点的存在
*/
int XmlNode::AssureChildExist(const char * nodename, int index)
{
	int ExistNumber = 0; /*已有的名字为nodename的子节点的个数*/
	if (NULL == nodename || index < 0)
	{
		return -1;
	}
	list<XmlNode*>::const_iterator it;
	for ( it = m_ChildList.begin(); it != m_ChildList.end(); ++it)
	{
		XmlNode * ptr = *it;
		if (ptr->GetNodeName() == nodename)
		{
			++ExistNumber;
			if (ExistNumber > index)
			{
				return 0;
			}
		}
	}
	for (;ExistNumber < (index+1); ++ExistNumber)
	{
		XmlNode * ptr = new XmlNode();
		if (NULL == ptr)
		{
			return -1;
		}
		ptr->SetNodeName(nodename);
		ptr->SetNodeValue("");
		ptr->SetNodeProperty("");
		ptr->SetParent(this);

		this->AddChild(ptr);
	}
	return 0;
}
