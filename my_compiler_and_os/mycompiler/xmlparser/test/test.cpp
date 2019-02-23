#include "xml.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main()
{
	    char errmsg[100];
       Xml xxx, xxx2;
       if (xxx.ReadFrmFile("./1.xml", errmsg, sizeof(errmsg)) != 0)
       {
               fprintf(stderr, "ReadFrmFile:[%s]\n", errmsg);
               return -1;
       }
	   char buf[1024];
	   memset(buf, 0, sizeof(buf));
	   xxx.WriteToBuffer(buf, sizeof(buf));
	   printf("buf=[%s]\n", buf);
	   /*
	   xxx.SetNodeInfo("/PayInst/RecvTreasuryCode", "<&>", "");
	   */
	   char val[100], prop[100];
	   xxx.GetNodeInfo("/PayInst/RecvTreasuryCode", val, sizeof(val), prop, sizeof(prop));
	   printf("\nval=[%s]\n", val);

	   xxx.WriteToFile("./2.xml", errmsg, sizeof(errmsg));
       return 0;
}

