#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xml.h"

#if !defined(_MAKE_DLL_)
int main(int argc, char** argv)
{
	if (argc < 2)
	{
		return 0;
	}
	Xml xxx;
	char errmsg[100];
	char buffer[1024];
	if (xxx.ReadFrmFile(argv[1], errmsg, sizeof(errmsg)) != 0)
	{
		printf("ERROR:[%s]\n", errmsg);
		return -1;
	}
	return 0;
}
#endif
