#include "btcp.h"
#include "circular_queue.h"
#include <poll.h>
#include <err.h>
#include <errno.h>


int btcp_tcpcli_connect(const char * ip, short int port, struct btcp_tcpconn_handler * handler)
{
    if (strlen(ip) >= INET_ADDRSTRLEN) {btcp_errno = ERR_INVALID_ARG; return -1;}
    memset(handler, 0, sizeof(struct btcp_tcpconn_handler));
    strcpy(handler->peer_ip, ip);
    int mtu = btcp_get_route_mtu(ip);
    printf("mtu for ip %s=%d\n", ip, mtu);
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
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(UDP_COMM_PORT);

        // 绑定套接字到指定端口
        if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
        {
            btcp_errno = ERR_INIT_UDP_FAIL;
            close(sockfd);
            return -1;
        }

        handler->udp_socket = sockfd;
    }
    // three handshakes
    {
        union btcp_tcphdr_with_option tcphdr;
        struct btcp_tcphdr * hdr = &tcphdr.base_hdr;
        memset(&tcphdr, 0, sizeof(union btcp_tcphdr_with_option));
        hdr->dest = htons(port);
        int localport = btcp_alloc_local_port();
        if (localport <= 0 || localport > 65535)
        {
            btcp_errno = ERR_GET_LPORT_FAIL;
            close(handler->udp_socket);
            return -1;
        }
        hdr->source = htons((short)localport);
        handler->local_port = localport;
        handler->peer_port = port;
        handler->local_seq = btcp_get_random();
        hdr->window = htons(DEF_RECV_BUFSZ);
        hdr->seq = htonl(handler->local_seq);
        btcp_set_tcphdr_flag(FLAG_SYN, &(hdr->doff_res_flags));

        int offset = sizeof(struct btcp_tcphdr);
        
        // mss
        *(uint8_t*)(tcphdr.options+ offset) = 0x02; offset+=1;
        *(uint8_t*)(tcphdr.options+ offset) = 0x04; offset+=1;
        *(uint16_t*)(tcphdr.options+ offset) = htons(handler->mss); offset+=2;

        

        struct sockaddr_in server_addr;
        // 初始化服务器地址结构
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = inet_addr(ip);
        server_addr.sin_port = htons(UDP_COMM_PORT);

        btcp_set_tcphdr_offset(offset, &hdr->doff_res_flags);
        
        if (sendto(handler->udp_socket, hdr, offset, 0, (const struct sockaddr*)&server_addr, sizeof(server_addr)) != offset)
        {
            btcp_errno = ERR_UDP_COMMM_FAIL;
            close(handler->udp_socket);
            return -1;
        }
        handler->status = SYNC_SENT;

        btcp_print_tcphdr((const char *)hdr, "send sync:");
    }
  
    return 0;
}
int btcp_handle_sync_sent(char * bigbuffer,  struct btcp_tcpconn_handler * handler)
{
    if (handler->status != SYNC_SENT)
    {
        
        return -1;
    }
    union btcp_tcphdr_with_option *tcphdr = (union btcp_tcphdr_with_option *)bigbuffer;
    struct btcp_tcphdr * hdr = &tcphdr->base_hdr;

    btcp_print_tcphdr((const char *)hdr, "recv ack:");

    if (!btcp_check_tcphdr_flag(FLAG_SYN, (hdr->doff_res_flags)) ||
        !btcp_check_tcphdr_flag(FLAG_ACK, (hdr->doff_res_flags))) 
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
    handler->peer_seq = ntohl(hdr->seq);
    uint32_t ack_seq = ntohl(hdr->ack_seq);
    if (ack_seq != (handler->local_seq + 1) )
    {
        btcp_errno = ERR_SEQ_MISMATCH;
        return -1;
    }
    handler->peer_recv_wnd_sz = ntohs(hdr->window);

    handler->local_seq++;
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
    hdr->dest = htons(handler->peer_port);
    hdr->source = htons(handler->local_port);
    hdr->seq = htonl(handler->local_seq);
    hdr->window = htons(DEF_RECV_BUFSZ);
    
    offset = sizeof(struct btcp_tcphdr);
    btcp_set_tcphdr_offset(offset, &hdr->doff_res_flags);

    struct sockaddr_in server_addr;
    // 初始化服务器地址结构
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(handler->peer_ip);
    server_addr.sin_port = htons(UDP_COMM_PORT);

    if (sendto(handler->udp_socket, hdr, offset, 0, (const struct sockaddr *)&server_addr, sizeof(server_addr)) != offset)
    {
        btcp_errno = ERR_UDP_COMMM_FAIL;
        close(handler->udp_socket);
        return -1;
    }
    handler->status = ESTABLISHED;

    btcp_print_tcphdr((const char *)hdr, "send ack:");
    return 0;
}

int main(int argc, char** argv)
{
    static struct btcp_tcpconn_handler handler;
    if (btcp_tcpcli_connect("192.168.0.11", 80, &handler))
    {
        printf("btcp_tcpcli_connect failed! %d\n", btcp_errno);
        return -1;
    }

    struct pollfd pfd[1];
    pfd[0].fd = handler.udp_socket;
    //pfd[0].events = POLLIN | POLLOUT;
    pfd[0].events = POLLIN;
    static char bigbuffer[1024*64] __attribute__((aligned(8))); 
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);

    while (1)
    {
        int ret = poll(pfd, 1, 1000); // 1 秒超时
        if (ret > 0)
        {
            if (pfd[0].revents & POLLIN)
            {
                
                ssize_t received = recvfrom(pfd[0].fd, bigbuffer, sizeof(bigbuffer), 0,
                                            (struct sockaddr *)&client_addr, &addr_len);
                if (received > 0)
                {
                    if (handler.status == SYNC_SENT)
                    {
                        btcp_handle_sync_sent(bigbuffer, &handler);
                    }
                }
                else if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    printf("No data available.\n");
                }
            }
            
        }
        else if (ret == 0)
        {
            printf("Timeout.\n");
        }
        else
        {
            perror("poll");
            break;
        }
    }
}

