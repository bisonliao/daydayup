#include "script.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <list>
#include "var.h"
#include "run.h"
using namespace std;
using namespace lnb;



int main(int argc, char**argv)
{
	if (argc < 2)
	{
		return 0;
	}
	FILE * fp = NULL;
	if ( (fp = fopen(argv[1], "rb")) == NULL)
	{
		perror("fopen:");
		return -1;
	}
	deque<string> PCodeList;
	while (1)
	{
		char buf[3000];

		if (fgets(buf, sizeof(buf)-1, fp) == NULL)
		{
			break;
		}
		int len = strlen(buf);
		if (len < 3)
		{
			fprintf(stderr, "·Ç·¨Ö¸Áî:%s\n", buf);
			return -1;
		}
		if (buf[len-1] == '\n')
		{
			buf[len-1] = '\0';
		}
		if (buf[len-2] == '\r')
		{
			buf[len-2] = '\0';
		}
		PCodeList.push_back(buf);
	}
	fclose(fp);
	fp = NULL;

	deque<string> args;
	return Run( PCodeList, args);
}
