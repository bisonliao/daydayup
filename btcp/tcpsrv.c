#include "btcp.h"
#include "circular_queue.h"
#include <poll.h>
#include <err.h>
#include <errno.h>


int btcp_tcpsrv_listen(const char * ip, short int port, struct btcp_tcpsrv_handler * srv)
{
    if (strlen(ip) >= INET_ADDRSTRLEN) {btcp_errno = ERR_INVALID_ARG; return -1;}
    memset(srv, 0, sizeof(struct btcp_tcpsrv_handler));
    strcpy(srv->my_ip, ip);
    srv->local_port = port;
    
    
    {
        int sockfd;
        struct sockaddr_in server_addr, client_addr;
        socklen_t addr_len = sizeof(client_addr);

        // 创建 UDP 套接字
        if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
        {
            btcp_errno = ERR_INIT_UDP_FAIL;
            return -1;
        }
        btcp_set_socket_nonblock(sockfd);

        // 初始化服务器地址结构
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = inet_addr(ip);
        server_addr.sin_port = htons(UDP_COMM_PORT);

        // 绑定套接字到指定端口
        if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
        {
            btcp_errno = ERR_INIT_UDP_FAIL;
            close(sockfd);
            return -1;
        }

        srv->udp_socket = sockfd;
    }
    return 0;
}  
 
int btcp_handle_sync_rcvd1(char * bigbuffer,  struct btcp_tcpconn_handler * handler, struct btcp_tcpsrv_handler* srv, const struct sockaddr_in * client_addr)
{
    
    union btcp_tcphdr_with_option *tcphdr = (union btcp_tcphdr_with_option *)bigbuffer;
    struct btcp_tcphdr * hdr = &tcphdr->base_hdr;

    btcp_print_tcphdr((const char *)hdr, "recv sync:");

    if (!btcp_check_tcphdr_flag(FLAG_SYN, (hdr->doff_res_flags)) ) 
    {
        btcp_errno = ERR_INVALID_PKG;
        return -1;
    }
    if ( ntohs(hdr->dest) != srv->local_port)
    {
        btcp_errno = ERR_PORT_MISMATCH;
        return -1;
    }
    
    
    {
        memset(handler, 0, sizeof(struct btcp_tcpconn_handler));

        
        
        handler->peer_port = ntohs(hdr->source);
        char peer_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr->sin_addr, peer_ip, INET_ADDRSTRLEN);

        printf("peer ip:%s\n", peer_ip);
        strcpy(handler->peer_ip, peer_ip);

        int mtu = btcp_get_route_mtu(peer_ip);
        if (mtu > 60)
        {
            handler->mss = mtu - 60;
        }
        else
        {
            btcp_errno = ERR_GET_MTU_FAIL;
            return -1;
        }
        if (btcp_init_queue(&handler->recv_buf, DEF_RECV_BUFSZ))
        {
            btcp_errno = ERR_INIT_CQ_FAIL;
            return -1;
        }
        if (btcp_init_queue(&handler->send_buf, DEF_SEND_BUFSZ))
        {
            btcp_errno = ERR_INIT_CQ_FAIL;
            return -1;
        }
        handler->cong_wnd = 1;
        handler->my_recv_wnd_sz = DEF_RECV_BUFSZ;

       
        handler->local_port = srv->local_port;
        
        handler->local_seq = btcp_get_random();
        
        handler->status = SYNC_RCVD;
        handler->udp_socket = srv->udp_socket;
    }

    handler->peer_seq = ntohl(hdr->seq);
    handler->peer_recv_wnd_sz = ntohs(hdr->window);

    int offset = sizeof(struct btcp_tcphdr);
    int hdrlen = btcp_get_tcphdr_offset(&hdr->doff_res_flags);
    while (offset < hdrlen )
    {
        uint8_t kind = *(uint8_t*)(tcphdr->options+offset);
        if (kind == 0x02) // mss
        {
            offset += 2;
            uint16_t mss = *(uint16_t*)(tcphdr->options+offset);
            offset += 2;
            mss = ntohs(mss);
            
            #ifdef _DEBUG_
            printf("peer mss:%d\n", mss);
            #endif
        }
    }

    //send ack package
    memset(tcphdr, 0, sizeof(union btcp_tcphdr_with_option));
    hdr->ack_seq = htonl(handler->peer_seq+1);
    btcp_set_tcphdr_flag(FLAG_ACK, &(hdr->doff_res_flags));
    printf("%u\n", hdr->doff_res_flags);
    btcp_set_tcphdr_flag(FLAG_SYN, &(hdr->doff_res_flags));
    printf("%u\n", hdr->doff_res_flags);
    hdr->dest = htons(handler->peer_port);
    hdr->source = htons(handler->local_port);
    hdr->seq = htonl(handler->local_seq);
    hdr->window = htons(DEF_RECV_BUFSZ);
    
    offset = sizeof(struct btcp_tcphdr);
    btcp_set_tcphdr_offset(offset, &hdr->doff_res_flags);
    printf("%u\n", hdr->doff_res_flags);

    struct sockaddr_in server_addr;
    // 初始化服务器地址结构
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(handler->peer_ip);
    server_addr.sin_port = htons(UDP_COMM_PORT);

    printf("send package to %s\n", handler->peer_ip);

    if (sendto(handler->udp_socket, hdr, offset, 0, (const struct sockaddr *)&server_addr, sizeof(server_addr)) != offset)
    {
        btcp_errno = ERR_UDP_COMMM_FAIL;
        close(handler->udp_socket);
        return -1;
    }
    handler->status = SYNC_RCVD;

    btcp_print_tcphdr((const char *)hdr, "send ack:");
    return 0;
}

int btcp_handle_sync_rcvd2(char * bigbuffer,  struct btcp_tcpconn_handler * handler, const struct sockaddr_in * client_addr)
{
    
    union btcp_tcphdr_with_option *tcphdr = (union btcp_tcphdr_with_option *)bigbuffer;
    struct btcp_tcphdr * hdr = &tcphdr->base_hdr;

    btcp_print_tcphdr((const char *)hdr, "recv ack:");

    if ( !btcp_check_tcphdr_flag(FLAG_SYN, (hdr->doff_res_flags)) ||
     !btcp_check_tcphdr_flag(FLAG_ACK, (hdr->doff_res_flags)) ) 
    {
        btcp_errno = ERR_INVALID_PKG;
        return -1;
    }
    
    if ( ntohs(hdr->source) != handler->peer_port ||
        ntohs(hdr->dest) != handler->local_port)
    {
        btcp_errno = ERR_PORT_MISMATCH;
        return -1;
    }
    uint32_t ack_seq = ntohl(hdr->ack_seq);
    if (ack_seq != (handler->local_seq + 1) )
    {
        btcp_errno = ERR_SEQ_MISMATCH;
        return -1;
    }
    handler->local_seq++;

    if (handler->peer_seq+1 !=(hdr->seq))
    {
        btcp_errno = ERR_SEQ_MISMATCH;
        return -1;
    }
    handler->peer_seq++;
    
    handler->status = ESTABLISHED;
    return 0;
}


static struct btcp_tcpconn_handler all_conn_handler[65536];

int main(int argc, char** argv)
{
    static struct btcp_tcpsrv_handler srv;
   
    
    if (btcp_tcpsrv_listen("192.168.0.11", 80, &srv) < 0)
    {
        printf("btcp_tcpsrv_listen failed, errno=%d\n", btcp_errno);
        return -1;
    }
    static char bigbuffer[100*1024]  __attribute__((aligned(8)));
    struct sockaddr_in client_addr;
    while (1)
    {
        int iret = btcp_is_readable(srv.udp_socket, 100, bigbuffer, sizeof(bigbuffer), &client_addr);
        if (iret > 0)
        {
            char ip_str[INET_ADDRSTRLEN]; // 用于存储IP地址字符串
            inet_ntop(AF_INET, &client_addr.sin_addr, ip_str, INET_ADDRSTRLEN);
            
            printf("recv %d bytes from %s\n", iret, ip_str);
            unsigned short dest_p, source_p;
            int dest_port = btcp_get_port(bigbuffer, &dest_p, &source_p);
            if (all_conn_handler[dest_p].local_port == 0) // not initialized
            {
                if (btcp_handle_sync_rcvd1(bigbuffer,  &all_conn_handler[dest_p], &srv, &client_addr))
                {
                    memset(&all_conn_handler[dest_p], 0, sizeof(struct btcp_tcpconn_handler)); // close the connn
                }
            }
            else if (all_conn_handler[dest_p].status == SYNC_RCVD)
            {
                if (btcp_handle_sync_rcvd2(bigbuffer,  &all_conn_handler[dest_p], &client_addr))
                {
                    memset(&all_conn_handler[dest_p], 0, sizeof(struct btcp_tcpconn_handler)); // close the connn
                }
            }


        }
        
        
    }

    
}

