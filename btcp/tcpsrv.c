#include "btcp.h"
#include "btcp_engine.h"

#include <poll.h>
#include <err.h>
#include <errno.h>




int main(int argc, char** argv)
{
    static struct btcp_tcpsrv_handler srv;
   
    
    if (btcp_tcpsrv_listen("192.168.0.11", 8080, &srv) < 0)
    {
        printf("btcp_tcpsrv_listen failed, errno=%d\n", btcp_errno);
        return -1;
    }
    btcp_tcpsrv_new_loop_thread(&srv);
    while (1)
    {
        GList *conns = btcp_tcpsrv_get_all_connections(&srv);
        if (conns != NULL)
        {

        }
        btcp_free_conns_in_glist(conns);
        usleep(1000000);
    }
    return 0;
    
    
}

