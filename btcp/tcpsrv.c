#include "btcp.h"
#include "btcp_engine.h"
#include "circular_queue.h"
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
    static char bigbuffer[100*1024]  __attribute__((aligned(8)));
    struct sockaddr_in client_addr;
    while (1)
    {
        int pkg_len = btcp_is_readable(srv.udp_socket, 100, bigbuffer, sizeof(bigbuffer), &client_addr);
        if (pkg_len > 0)
        {
            char ip_str[INET_ADDRSTRLEN]; // 用于存储IP地址字符串
            inet_ntop(AF_INET, &client_addr.sin_addr, ip_str, INET_ADDRSTRLEN);
            
            
            unsigned short dest_p, source_p;
            btcp_get_port(bigbuffer, &dest_p, &source_p);
            g_info("rcvd pkg, source:%d, peer_port:%d, status:%d", source_p, all_conn_handler[source_p].peer_port, all_conn_handler[source_p].status);
            if (all_conn_handler[source_p].peer_port == 0) // not initialized
            {
                if (btcp_handle_sync_rcvd1(bigbuffer,  &all_conn_handler[source_p], &srv, &client_addr))
                {
                    fprintf(stderr, "btcp_handle_sync_rcvd1() failed! %d\n", btcp_errno);
                    memset(&all_conn_handler[dest_p], 0, sizeof(struct btcp_tcpconn_handler)); // close the connn
                }
            }
            else if (all_conn_handler[source_p].status == SYNC_RCVD)
            {
                if (btcp_handle_sync_rcvd2(bigbuffer,  &all_conn_handler[source_p], &client_addr))
                {
                    fprintf(stderr, "btcp_handle_sync_rcvd2() failed! %d\n", btcp_errno);
                    memset(&all_conn_handler[dest_p], 0, sizeof(struct btcp_tcpconn_handler)); // close the connn
                }
            }
            else if (all_conn_handler[source_p].status == ESTABLISHED)
            {
                if (btcp_handle_data_rcvd(bigbuffer, pkg_len, &all_conn_handler[source_p], &client_addr))
                {
                    fprintf(stderr, "btcp_handle_data_rcvd() failed! %d\n", btcp_errno);
                    memset(&all_conn_handler[dest_p], 0, sizeof(struct btcp_tcpconn_handler)); // close the connn
                }
            }


        }
        
        
    }

    
}

