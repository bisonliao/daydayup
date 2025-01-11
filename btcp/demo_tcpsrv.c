#include "btcp_api.h"


#include <poll.h>
#include <err.h>
#include <errno.h>
#include  <glib.h>

/*
 * 正如文件名字所说，这是一个 tcp server 的用户程序 demo
 */


int main(int argc, char** argv)
{
    static struct btcp_tcpsrv_handler srv;
   
    
    if (btcp_tcpsrv_listen("192.168.0.11", 8080, &srv) < 0)
    {
        printf("btcp_tcpsrv_listen failed, errno=%d\n", btcp_errno);
        return -1;
    }
    btcp_tcpsrv_new_loop_thread(&srv);
    static char bigbuffer[100*1024] __attribute__((aligned(8))); // 用于临时收发包，不会跨线程也不会跨连接使用
    uint64_t total = 0;
    char last_char = 0;
    while (1)
    {
        //int status = ESTABLISHED;
        GList *conns = btcp_tcpsrv_get_all_conn_fds(&srv, NULL); 
        if (conns != NULL)
        {
            static struct pollfd pfd[MAX_CONN_ALLOWED];
            int i;
            GList * iter;
            for (iter = conns, i=0; iter != NULL && i < MAX_CONN_ALLOWED; iter = iter->next, i++)
            {
                int fd  = GPOINTER_TO_INT(iter->data);
                pfd[i].fd = fd;
                pfd[i].events = POLLIN;
              
            }
            int fd_num = i;
            
            
            int ret = poll(pfd, fd_num, 100); 
            
            if (ret > 0)
            {
                for (i = 0; i < fd_num; ++i)
                {
                    if (pfd[i].revents & POLLIN) 
                    {

                        ssize_t received = read(pfd[i].fd, bigbuffer, sizeof(bigbuffer));
                        g_info("recv remote data, len=%d", received);
                        if (received > 0)
                        {
                            bigbuffer[received] = 0;
                            printf("[%s]\n", bigbuffer);
                            total += received;
                            printf(">>>>>>>>>total=%llu\n", total);
                            if (last_char != 0 && last_char != 'z')
                            {
                                g_assert (last_char+1 == bigbuffer[0]);
                                last_char = bigbuffer[received-1];
                            }
                        }
                        else if (received == 0)
                        {
                            g_info("detect remote closed, I close the conn too. %d", pfd[i].fd);
                            close(pfd[i].fd);
                        }
                        else 
                        {
                            if (errno == EAGAIN || errno == EWOULDBLOCK){
                                printf("No data available.\n");
                            }
                        }
                    }
                }
            }

            btcp_free_conns_in_glist(conns);
            conns = NULL;
        }
        else
        {
            //printf("no established conns\n");
        }
        
    }
    return 0;
    
    
}

