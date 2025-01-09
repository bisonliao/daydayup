#ifndef BTCP_SELECTIVE_ACK_BLOCKLIST_H_INCLUDED
#define BTCP_SELECTIVE_ACK_BLOCKLIST_H_INCLUDED

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <glib.h>
#include "tool.h"



struct btcp_sack_blocklist
{
    GList *blocklist;
};

int btcp_sack_blocklist_init(struct btcp_sack_blocklist *list);
int btcp_sack_blocklist_add_record(struct btcp_sack_blocklist *list , const struct btcp_range * range);
int btcp_sack_blocklist_destroy(struct btcp_sack_blocklist *list);




#endif