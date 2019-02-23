#if !defined(__XML_NODE_H_INCLUDED__)
#define __XML_NODE_H_INCLUDED__

#include <list>
#include "AnsiString.h"

using namespace std;


#define PXMLNODE_LIST  list<XmlNode*> 

class XmlNode
{

public:
	XmlNode();
	~XmlNode();
	const AnsiString & GetNodeName() const;
	const AnsiString & GetNodeProperty() const;
	const AnsiString & GetNodeValue() const;
	const PXMLNODE_LIST & GetNodeChildList() const;
	XmlNode * GetParent() const;
	void  SetNodeName(const AnsiString & name);
	void  SetNodeValue(const AnsiString & value);
	void  SetNodeProperty(const AnsiString & property);
	void  SetParent(XmlNode* parent);
	int  AddChild(XmlNode * pNode);
	/*
	*取第index个名字为nodename的子节点的指针
	*/
	XmlNode * GetChildByNodeName(const char * nodename, int index) const;

	/*
	* 确保第index个名字为nodename的子节点的存在
	*/
	int AssureChildExist(const char * nodename, int index);

private:
	/*被禁止使用的成员函数*/
	XmlNode(const XmlNode& another);
	bool operator==(const XmlNode& another);
	const XmlNode& operator=(const XmlNode& another);


	/*
	*	例如对于 <student age='3'>小明</student>
	*	m_NodeName="student"
	*	m_NodeProperty="age='3'"
	*	m_NodeValue="小明"
	*	m_Child为空
	*/
	AnsiString m_NodeName;		/*字段的名称*/
	AnsiString m_NodeProperty;	/*字段的属性字符串*/
	AnsiString m_NodeValue;		/*字段的值*/


	PXMLNODE_LIST m_ChildList;		/*下一级节点的指针列表*/
	XmlNode*	m_Parent;			/*父节点指针*/
};

#endif
