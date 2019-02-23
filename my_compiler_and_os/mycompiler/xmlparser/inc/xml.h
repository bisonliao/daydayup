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
	*从文件或者缓冲区内存读入一棵xml树
	*/
	int ReadFrmFile(const char * xmlfilename, char* errmsg, unsigned short errmsgsize);
	int ReadFrmBuffer(const char * buffer, unsigned short buffersize, char *errmsg, unsigned short errmsgsize);

	/*
	*将xml树输出到一个文件或者内存
	*/
	int WriteToFile(const char * xmlfilename, char* errmsg, unsigned short errmsgsize);
	int WriteToBuffer(char * buffer, unsigned short buffersize);

	/*
	*得到xml根节点的名字
	*/
	const char * GetRootName() const;
	/*
	*清除xml树, 释放内存
	*/
	void Clear();

	/*
	*读取路径path指定的节点的属性和值 例如path="/FileTransfer", path="/FileTransfer/filename"
	*/
	int GetNodeInfo(const char * path, char* NodeValue, unsigned int NodeValueSize,
									   char* NodeProperty, unsigned int NodePropertySize);

	/*
	*设置路径path指定的节点的属性和值
	*/
	int SetNodeInfo(const char * path, const char* NodeValue, const char* NodeProperty);

	/*
	* 初始化根节点
	*/
	int InitRoot(const char* RootName, const char* RootValue, const char* RootProperty);

	/*
	* 将一个节点以下的xml树分支全部导出 bNodeSelf标识是否导出这个节点本身
	*/
	int ExportNode(const char* path, char* buffer, int buffersize, bool bNodeSelf );


private:
	/*被禁止使用的成员函数*/
	Xml(const Xml& another);
	bool operator==(const Xml& another);
	const Xml& operator=(const Xml& another);

	XmlNode* m_RootPtr;		/*根指针*/

	/*对所有节点进行深度优先遍历，得到节点指针的列表*/
	int DepthScan(list<XmlNode*> &lst) const;

	/*分析路径*/
	static int SplitPath(const AnsiString & path, list<AnsiString> & lst1, list<int> &lst2);

	/*转义*/
	static void TransferMeaning(AnsiString & xxx);
	static void TransferMeaning2(AnsiString & xxx);

	/*字符串和属性列表的相互转化*/
	//static int String2PropertyList(const AnsiString & str, 
};

#endif
