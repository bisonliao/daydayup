#include "btcp.h"
#include "circular_queue.h"
#include <poll.h>
#include <err.h>
#include <errno.h>
#include <pthread.h>
#include "btcp_range.h"
#include <glib.h>

int btcp_tcpcli_init_udp(struct btcp_tcpconn_handler * handler) // 创建背后通信用的udp套接字
{
    if (handler == NULL) { btcp_errno = ERR_INVALID_ARG; return -1;}
    
    memset(handler, 0, sizeof(struct btcp_tcpconn_handler));

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
    server_addr.sin_port = htons(0);

    // 绑定套接字到指定端口
    if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        btcp_errno = ERR_INIT_UDP_FAIL;
        close(sockfd);
        return -1;
    }
    // 获取系统自动分配的端口号
    addr_len = sizeof(server_addr);
    if (getsockname(sockfd, (struct sockaddr *)&server_addr, &addr_len) == -1) {
        btcp_errno = ERR_INIT_UDP_FAIL;
        close(sockfd);
        return -1;
    }
    handler->local_port = ntohs(server_addr.sin_port);
    handler->udp_socket = sockfd;
    return 0;
}


int btcp_tcpcli_connect(const char * ip, short int port, struct btcp_tcpconn_handler * handler)
{
    if (strlen(ip) >= INET_ADDRSTRLEN) {btcp_errno = ERR_INVALID_ARG; return -1;}
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
    if (btcp_send_queue_init(&handler->send_buf, DEF_SEND_BUFSZ))
    {
        btcp_errno = ERR_INIT_CQ_FAIL; 
        return -1;
    }
    handler->cong_wnd = 1;
    handler->cong_wnd_threshold = 8;
    handler->my_recv_wnd_sz = DEF_RECV_BUFSZ;
    
    if (btcp_tcpcli_init_udp(&handler)) { return -1;}

    // 创建一对已连接的 Unix Domain Socket
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, &handler->user_socket_pair) == -1) {
        perror("socketpair");
        exit(EXIT_FAILURE);
    }

    // three handshakes
    {
        union btcp_tcphdr_with_option tcphdr;
        struct btcp_tcphdr * hdr = &tcphdr.base_hdr;
        memset(&tcphdr, 0, sizeof(union btcp_tcphdr_with_option));
        hdr->dest = htons(port);
        
        hdr->source = htons(handler->local_port);
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
        server_addr.sin_port = htons(port);

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
    server_addr.sin_port = htons(handler->peer_port);

    if (sendto(handler->udp_socket, hdr, offset, 0, (const struct sockaddr *)&server_addr, sizeof(server_addr)) != offset)
    {
        btcp_errno = ERR_UDP_COMMM_FAIL;
        close(handler->udp_socket);
        return -1;
    }
    handler->status = ESTABLISHED;

    btcp_print_tcphdr((const char *)hdr, "send ack:");
    printf("established!\n");
    return 0;
}

int btcp_try_send(struct btcp_tcpconn_handler *handler)
{
    int retcode = -1;
    //计算发送窗口大小，单位为byte
    int send_wndsz = handler->cong_wnd * handler->mss;
    if (send_wndsz > handler->peer_recv_wnd_sz)
    {
        send_wndsz = handler->peer_recv_wnd_sz;
    }
    if (send_wndsz < 1)
    {
        return 0;
    }
    //发送窗口的范围，与已经发送的待ack报文覆盖的范围比较，找出需要发送的数据段 ，
    //这里参与运算的seq/from/to使用uint64_t类型，且保证to >= from，即to可能大于UINT32_MAX
    struct btcp_range* range_to_send = malloc(sizeof(struct btcp_range));
    range_to_send->from = handler->send_buf.start_seq;
    range_to_send->to = (uint64_t)(handler->send_buf.start_seq) + send_wndsz - 1; //闭区间，所以要减一

    GList *range_list_to_send = NULL;
    range_list_to_send = g_list_append(NULL, range_to_send);
    
    GList *range_list_sent = NULL;
    if (btcp_timeout_get_all_event(&handler->timeout, &range_list_sent) != 0)
    {
        btcp_errno = ERR_MEM_ERROR;
        goto btcp_try_send_out;
    }
    GList * range_list_result = NULL, *combined_list = NULL;
    if (btcp_range_subtract(range_list_to_send, range_list_sent, &range_list_result))
    {
        btcp_errno = ERR_MEM_ERROR;
        goto btcp_try_send_out;
    }
    btcp_range_list_combine(range_list_result, &combined_list);
    //发送，并插入超时等待队列
    struct sockaddr_in server_addr;
    // 初始化服务器地址结构
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(handler->peer_ip);
    server_addr.sin_port = htons(handler->peer_port);
    for (const GList *iter = combined_list; iter != NULL; iter = iter->next)
    {
        struct btcp_range *a_range = (struct btcp_range *)iter->data;
        struct btcp_range b_range;
        b_range.from = a_range->from;
        b_range.to = a_range->to;
        static unsigned char bigbuffer[100*1024];
        while (b_range.from <= b_range.to) // 如果超过mss，需要发送多次
        {
            int datalen = b_range.to - b_range.from + 1;
            if (datalen > handler->mss)
            {
                datalen = handler->mss;
            }
            btcp_send_queue_fetch_data(&handler->send_buf, b_range.from, b_range.from + datalen - 1, bigbuffer+sizeof(struct btcp_tcphdr));

            
            struct btcp_tcphdr * hdr = bigbuffer;
            memset(&hdr, 0, sizeof(struct btcp_tcphdr));
            hdr->dest = htons(handler->peer_port);
            hdr->source = htons(handler->local_port);
            hdr->window = htons(DEF_RECV_BUFSZ);
            hdr->seq = htonl(b_range.from % ((uint64_t)1 + UINT32_MAX));
            int offset = sizeof(struct btcp_tcphdr);
            btcp_set_tcphdr_offset(offset, &hdr->doff_res_flags);
            offset += datalen;

            int sent_len = sendto(handler->udp_socket, hdr, offset, 0, (const struct sockaddr *)&server_addr, sizeof(server_addr));
            if (sent_len < 0) // udp发包，不存在只发部分报文的情况，要么完整报文，要么负1
            {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    retcode = 0;
                    goto btcp_try_send_out;
                }
                
                btcp_errno = ERR_UDP_COMMM_FAIL;
                close(handler->udp_socket);
                goto btcp_try_send_out;
            }
            // 记录超时事件
            struct btcp_range c_range;
            c_range.from = b_range.from % UINT16_MAX;
            c_range.to = b_range.to % UINT16_MAX;

            btcp_timeout_add_event(&handler->timeout, 5, &c_range, sizeof(c_range), btcp_range_cmp);

            b_range.from += datalen;
        }
    }

    retcode = 0;
btcp_try_send_out:
    btcp_range_free_list(range_list_to_send);
    btcp_range_free_list(range_list_sent);
    btcp_range_free_list(range_list_result);
    btcp_range_free_list(combined_list);

    return retcode;
}

int btcp_check_send_timeout(struct btcp_tcpconn_handler *handler)
{
    struct btcp_range e;
    int len = sizeof(struct btcp_range);
    int timeout_occur = 0;
    while ( btcp_timeout_check(&handler->timeout, &e, &len) == 1)
    {
        timeout_occur = 1; //有超时发生
    }
    if (timeout_occur)
    {
        // 修改发送窗口大小
        handler->cong_wnd_threshold = handler->cong_wnd / 2;
        handler->cong_wnd = 1;
        if (handler->cong_wnd_threshold < 4)
        {
            handler->cong_wnd_threshold = 4;
        }
    }
    return timeout_occur;
}


int btcp_tcpcli_loop(struct btcp_tcpconn_handler *handler)
{
    int timeout = 100; // 默认0.1s
    
    struct pollfd pfd[2];
    pfd[0].fd = handler->udp_socket;
    pfd[0].events = POLLIN;

    pfd[1].fd = handler->user_socket_pair[1];
    pfd[1].events = POLLIN;

    static char bigbuffer[1024*64] __attribute__((aligned(8))); 
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);

    while (1)
    {
        int ret = poll(pfd, 2, timeout); // 1 秒超时
        if (ret > 0)
        {
            if (pfd[0].revents & POLLIN) //底层udp可读，收包并处理
            {
                
                ssize_t received = recvfrom(pfd[0].fd, bigbuffer, sizeof(bigbuffer), 0,
                                            (struct sockaddr *)&client_addr, &addr_len);
                if (received > 0)
                {
                    if (handler->status == SYNC_SENT)
                    {
                        btcp_handle_sync_sent(bigbuffer, handler);
                    }
                    else if (handler->status == ESTABLISHED)
                    {
                        btcp_handle_data_rcvd(bigbuffer, handler);
                    }

                }
                else if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    printf("No data available.\n");
                }
            }
            if (pfd[1].revents & POLLIN) //用户层发数据过来了，放置到发送队列里
            {
                int space = btcp_send_queue_get_available_space(&handler->send_buf); // 获得发送缓冲区的空闲空间大小
                if (space > 0)
                {
                    ssize_t received = read(pfd[1].fd, bigbuffer, space);
                    if (received > 0)
                    {
                        btcp_send_queue_enqueue(&handler->send_buf, bigbuffer, received);
                        btcp_try_send(&handler);
                    }
                }
                
            }
            
        }
        else if (ret == 0)
        {
            //printf("Timeout.\n");
        }
        else
        {
            perror("poll");
            break;
        }
        
        if (btcp_check_send_timeout(&handler))//检查可能的发包超时未ack
        {
            //btcp_try_send(&handler); // 立即（重）发tcp包，因为下面本身也会调用，所以先注释掉
        }
        btcp_try_send(&handler); // 尝试发tcp包
        if (btcp_send_queue_have_data_to_send(&handler))
        {
            // 只要还有tcp报文未发送，那么超时时间就极短
            timeout = 0;
        }
        else
        {
            timeout = 100;
        }
    }
}

int btcp_tcpcli_new_loop_thread(struct btcp_tcpconn_handler *handler)
{
    pthread_t thread_id;
    int arg = 42; // 传递给线程的参数
    void *retval; // 用于存储线程的返回值

    // 创建线程
    if (pthread_create(&thread_id, NULL, btcp_tcpcli_loop, (void *)handler) != 0) {
        perror("pthread_create");
        return -1;
    }
    if (pthread_detach(thread_id))
    {
        perror("pthread_detach");
        return -1;
    }

    return 0;
}

int main(int argc, char** argv)
{
    static struct btcp_tcpconn_handler handler;
    if (btcp_tcpcli_init(&handler))
    {
        printf("btcp_tcpcli_init failed! %d\n", btcp_errno);
        return -1;
    }
    if (btcp_tcpcli_connect("192.168.0.11", 80, &handler))
    {
        printf("btcp_tcpcli_connect failed! %d\n", btcp_errno);
        return -1;
    }
    btcp_tcpcli_new_loop_thread(&handler);
    while (1)
    {
        char buf[1024];
        ssize_t sz = read(0, buf, sizeof(buf));
        write(handler.user_socket_pair[0], buf, sz);
    }
    return 0;
}

