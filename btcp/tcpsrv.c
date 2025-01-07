#include "btcp.h"
#include "btcp_engine.h"

#include <poll.h>
#include <err.h>
#include <errno.h>
#include  <glib.h>




int main(int argc, char** argv)
{
    static struct btcp_tcpsrv_handler srv;
   
    
    if (btcp_tcpsrv_listen("192.168.0.11", 8080, &srv) < 0)
    {
        printf("btcp_tcpsrv_listen failed, errno=%d\n", btcp_errno);
        return -1;
    }
    btcp_tcpsrv_new_loop_thread(&srv);
    static char bigbuffer[100*1024];
    while (1)
    {
        GList *conns = btcp_tcpsrv_get_all_connections(&srv);
        if (conns != NULL)
        {
            struct pollfd pfd[1024];
            int i;
            GList * iter;
            for (iter = conns, i=0; iter != NULL && i < 1024; iter = iter->next, i++)
            {
                const struct btcp_tcpconn_handler * handler = (const struct btcp_tcpconn_handler *)(iter->data);
                g_info("in %s, socketpair:%d, %d", __FILE__,
                        handler->user_socket_pair[0], handler->user_socket_pair[1]);
                pfd[i].fd = handler->user_socket_pair[0];
                pfd[i].events = POLLIN;
            }
            int fd_num = i;
            
            
            int ret = poll(pfd, fd_num, 100); // 1 秒超时
            printf("fd num:%d, poll return %d\n", fd_num, ret);
            if (ret > 0)
            {
                for (i = 0; i < fd_num; ++i)
                {
                    if (pfd[i].revents & POLLIN) 
                    {

                        ssize_t received = read(pfd[i].fd, bigbuffer, sizeof(bigbuffer));
                        g_info("recv remote data, len=%d\n", received);
                        if (received > 0)
                        {
                            bigbuffer[received] = 0;
                            printf("%s", bigbuffer);
                        }
                        else if (errno == EAGAIN || errno == EWOULDBLOCK)
                        {
                            printf("No data available.\n");
                        }
                    }
                }
            }
            btcp_free_conns_in_glist(conns);
            conns = NULL;
        }
        else
        {
            printf("no established conns\n");
        }
        
        usleep(1000000);
    }
    return 0;
    
    
}

