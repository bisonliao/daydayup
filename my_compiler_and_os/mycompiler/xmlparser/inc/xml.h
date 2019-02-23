#if !defined(__XML_H_INCLUDED__)
#define  __XML_H_INCLUDED__

#include <list>

#include "xmlnode.h"

using namespace std;

#if defined(_WIN32_)
class __declspec(dllexport) Xml
//class Xml
#else
class Xml
#endif
{
public:
	Xml();
	~Xml();
	/*
	*���ļ����߻������ڴ����һ��xml��
	*/
	int ReadFrmFile(const char * xmlfilename, char* errmsg, unsigned short errmsgsize);
	int ReadFrmBuffer(const char * buffer, unsigned short buffersize, char *errmsg, unsigned short errmsgsize);

	/*
	*��xml�������һ���ļ������ڴ�
	*/
	int WriteToFile(const char * xmlfilename, char* errmsg, unsigned short errmsgsize);
	int WriteToBuffer(char * buffer, unsigned short buffersize);

	/*
	*�õ�xml���ڵ������
	*/
	const char * GetRootName() const;
	/*
	*���xml��, �ͷ��ڴ�
	*/
	void Clear();

	/*
	*��ȡ·��pathָ���Ľڵ�����Ժ�ֵ ����path="/FileTransfer", path="/FileTransfer/filename"
	*/
	int GetNodeInfo(const char * path, char* NodeValue, unsigned int NodeValueSize,
									   char* NodeProperty, unsigned int NodePropertySize);

	/*
	*����·��pathָ���Ľڵ�����Ժ�ֵ
	*/
	int SetNodeInfo(const char * path, const char* NodeValue, const char* NodeProperty);

	/*
	* ��ʼ�����ڵ�
	*/
	int InitRoot(const char* RootName, const char* RootValue, const char* RootProperty);

	/*
	* ��һ���ڵ����µ�xml����֧ȫ������ bNodeSelf��ʶ�Ƿ񵼳�����ڵ㱾��
	*/
	int ExportNode(const char* path, char* buffer, int buffersize, bool bNodeSelf );


private:
	/*����ֹʹ�õĳ�Ա����*/
	Xml(const Xml& another);
	bool operator==(const Xml& another);
	const Xml& operator=(const Xml& another);

	XmlNode* m_RootPtr;		/*��ָ��*/

	/*�����нڵ����������ȱ������õ��ڵ�ָ����б�*/
	int DepthScan(list<XmlNode*> &lst) const;

	/*����·��*/
	static int SplitPath(const AnsiString & path, list<AnsiString> & lst1, list<int> &lst2);

	/*ת��*/
	static void TransferMeaning(AnsiString & xxx);
	static void TransferMeaning2(AnsiString & xxx);

	/*�ַ����������б���໥ת��*/
	//static int String2PropertyList(const AnsiString & str, 
};

#endif
