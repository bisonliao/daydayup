#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "token.h"
#include <vector>
#include <deque>
using namespace std;
using namespace lnb;

#include "compile.h"

int main(int argc, char ** argv)
{
	if (argc != 2)
	{
		fprintf(stderr, "usage: %s source_file\n", argv[0]);
		return -1;
	}
	deque<string>  PCodeList;
	if ( compile(argv[1], PCodeList) == 0)
	{
		deque<string>::const_iterator it;
		for (it = PCodeList.begin(); it != PCodeList.end(); ++it)
		{
			printf("%s\n", it->c_str());
		}
	}
	return 0;
}
