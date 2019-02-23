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
	*ȡ��index������Ϊnodename���ӽڵ��ָ��
	*/
	XmlNode * GetChildByNodeName(const char * nodename, int index) const;

	/*
	* ȷ����index������Ϊnodename���ӽڵ�Ĵ���
	*/
	int AssureChildExist(const char * nodename, int index);

private:
	/*����ֹʹ�õĳ�Ա����*/
	XmlNode(const XmlNode& another);
	bool operator==(const XmlNode& another);
	const XmlNode& operator=(const XmlNode& another);


	/*
	*	������� <student age='3'>С��</student>
	*	m_NodeName="student"
	*	m_NodeProperty="age='3'"
	*	m_NodeValue="С��"
	*	m_ChildΪ��
	*/
	AnsiString m_NodeName;		/*�ֶε�����*/
	AnsiString m_NodeProperty;	/*�ֶε������ַ���*/
	AnsiString m_NodeValue;		/*�ֶε�ֵ*/


	PXMLNODE_LIST m_ChildList;		/*��һ���ڵ��ָ���б�*/
	XmlNode*	m_Parent;			/*���ڵ�ָ��*/
};

#endif
