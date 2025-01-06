#include "btcp.h"
#include "circular_queue.h"
#include <poll.h>
#include <err.h>
#include <errno.h>
#include <pthread.h>
#include "btcp_range.h"
#include <glib.h>
#include <unistd.h>


int main(int argc, char** argv)
{
    static struct btcp_tcpconn_handler handler;
#if 0
    g_log_set_handler(NULL, G_LOG_LEVEL_WARNING | G_LOG_LEVEL_ERROR | G_LOG_LEVEL_CRITICAL|G_LOG_LEVEL_INFO,
                      g_log_default_handler, NULL);
#endif
    
    if (btcp_tcpcli_connect("192.168.0.11", 8080, &handler))
    {
        printf("btcp_tcpcli_connect failed! %d\n", btcp_errno);
        return -1;
    }
    g_info("in main(), peer ip:%s, mss:%d, peer_port:%d\n", handler.peer_ip, handler.mss, handler.peer_port);
    btcp_tcpcli_new_loop_thread(&handler);
    while (1)
    {
        if (handler.status != ESTABLISHED)
        {
            printf("waiting...\n");
            usleep(1000);
            continue;
        }
        char buf[1024];
        ssize_t sz = 100;
        //sz = read(0, buf, sizeof(buf));
        memset(buf, 'A', sz);
        int iret = write(handler.user_socket_pair[0], buf, sz);
        //printf("write %d bytes into engine, %u\n", iret, &handler);
        usleep(1000000);
    }
    return 0;
}

