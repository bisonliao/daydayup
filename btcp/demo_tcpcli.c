#include "btcp_api.h"
#include <poll.h>
#include <err.h>
#include <errno.h>
#include <pthread.h>
#include <glib.h>
#include <unistd.h>
/*
 * 正如文件名字所说，这是一个 tcp client的用户程序 demo
 */


int main(int argc, char** argv)
{
    static struct btcp_tcpconn_handler handler;
    
    if (btcp_tcpcli_connect("192.168.0.11", 8080, &handler))
    {
        g_warning("btcp_tcpcli_connect failed! %d", btcp_errno);
        return -1;
    }
    g_info("in main(), peer ip:%s, mss:%d, peer_port:%d\n", handler.peer_ip, handler.mss, handler.peer_port);
    
    btcp_tcpcli_new_loop_thread(&handler); // 启动tcp引擎的工作线程
    
    uint64_t total = 0; // 统计用户层发送了多少字节
    while (total < 10000)
    {
        if (handler.status != ESTABLISHED)
        {
            usleep(1000);
            continue;
        }

        // 准备 26个英文小写字母
        char buf[1024];
        ssize_t sz = 26;
        
        for (int i = 0; i < sz; ++i)
        {
            buf[i] = 'a'+i;
        }
        int offset = 0;
       
        // 不断的通过tcp连接发送给服务器
        while (1)
        {
            int iret = write(handler.user_socket_pair[0], buf+offset, sz-offset);
            if (iret <0 )
            {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    usleep(100);
                    continue;
                }
                else
                {
                    perror("write");
                    return -1;
                }
            }
            else 
            {
                total += iret;
                offset += iret;
                g_info(">>>>>>>>>total=%llu", total);
                if (offset == sz)
                {
                    break;
                }
    
            }
            
        }
       
        usleep(1000000*3);
        
    }
    g_info("client close the conn");
    close(handler.user_socket_pair[0]);
    while (1)
    {
        usleep(2000000);
        g_info("status=%d", handler.status);
    }
   
    return 0;
}

