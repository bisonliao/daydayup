#include "xml.h"

#define DLL_SPEC extern "C" __declspec(dllexport) 

DLL_SPEC unsigned int xml_open()
{
	Xml *pXml = NULL;
	pXml = new Xml();
	return (unsigned int)pXml;
}
DLL_SPEC int xml_ReadFrmFile(unsigned int handle,
		const char * xmlfilename, 
		char* errmsg, 
		unsigned short errmsgsize)
{
	if (handle == 0)
	{
		_snprintf(errmsg, errmsgsize, "句柄无效");
		return -1;
	}
	Xml *pXml = (Xml*)handle;
	return pXml->ReadFrmFile(xmlfilename, errmsg, errmsgsize);
}
DLL_SPEC int xml_WriteToFile(unsigned int handle,
	const char * xmlfilename, 
	char* errmsg, 
	unsigned short errmsgsize)
{
	if (handle == 0)
	{
		_snprintf(errmsg, errmsgsize, "句柄无效");
		return -1;
	}
	Xml *pXml = (Xml*)handle;
	return pXml->WriteToFile(xmlfilename, errmsg, errmsgsize);
}
DLL_SPEC int xml_ReadFrmBuffer(unsigned int handle,
	const char * buffer, 
	unsigned short buffersize, 
	char *errmsg, 
	unsigned short errmsgsize)
{
	if (handle == 0)
	{
		_snprintf(errmsg, errmsgsize, "句柄无效");
		return -1;
	}
	Xml *pXml = (Xml*)handle;
	return pXml->ReadFrmBuffer(buffer, buffersize, errmsg, errmsgsize);
}
DLL_SPEC int xml_WriteToBuffer(unsigned int handle,
		char * buffer, unsigned short buffersize)
{
	if (handle == 0)
	{
		//_snprintf(errmsg, errmsgsize, "句柄无效");
		return -1;
	}
	Xml *pXml = (Xml*)handle;
	return pXml->WriteToBuffer(buffer, buffersize);
}
DLL_SPEC const char * xml_GetRootName(unsigned int handle)
{
	if (handle == 0)
	{
		//_snprintf(errmsg, errmsgsize, "句柄无效");
		return NULL;
	}
	Xml *pXml = (Xml*)handle;
	return pXml->GetRootName();
}
DLL_SPEC void xml_Clear(unsigned int handle)
{
	if (handle == 0)
	{
		//_snprintf(errmsg, errmsgsize, "句柄无效");
		return ;
	}
	Xml *pXml = (Xml*)handle;
	return pXml->Clear();
}
DLL_SPEC int xml_GetNodeInfo(unsigned int handle,
			const char * path, char* NodeValue, unsigned int NodeValueSize,
			char* NodeProperty, unsigned int NodePropertySize)
{
	if (handle == 0)
	{
		//_snprintf(errmsg, errmsgsize, "句柄无效");
		return -1;
	}
	Xml *pXml = (Xml*)handle;
	return pXml->GetNodeInfo(path, NodeValue, NodeValueSize, NodeProperty, NodePropertySize);
}
DLL_SPEC int xml_SetNodeInfo(unsigned int handle,
				const char * path, const char* NodeValue, const char* NodeProperty)
{
	if (handle == 0)
	{
		//_snprintf(errmsg, errmsgsize, "句柄无效");
		return -1;
	}
	Xml *pXml = (Xml*)handle;
	return pXml->SetNodeInfo(path, NodeValue, NodeProperty);
}
DLL_SPEC int xml_InitRoot(unsigned int handle,
			const char* RootName, const char* RootValue, const char* RootProperty)
{
	if (handle == 0)
	{
		//_snprintf(errmsg, errmsgsize, "句柄无效");
		return -1;
	}
	Xml *pXml = (Xml*)handle;
	return pXml->InitRoot(RootName, RootValue, RootProperty);
}
DLL_SPEC int xml_ExportNode(unsigned int handle,
	const char* path, char* buffer, int buffersize, bool bNodeSelf )
{
	if (handle == 0)
	{
		//_snprintf(errmsg, errmsgsize, "句柄无效");
		return -1;
	}
	Xml *pXml = (Xml*)handle;
	return pXml->ExportNode(path, buffer, buffersize, bNodeSelf);
}
DLL_SPEC void xml_close(unsigned int handle)
{
	if (handle == 0)
	{
		//_snprintf(errmsg, errmsgsize, "句柄无效");
		return;
	}
	Xml *pXml = (Xml*)handle;
	delete pXml;
}