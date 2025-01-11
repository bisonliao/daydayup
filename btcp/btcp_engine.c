#include "btcp_api.h"
#include "btcp_engine.h"
#include <poll.h>
#include <stdint.h>
/*
 * tcp 引擎的实现， 主要形式是一个死循环的工作线程
 * 它不断的从底层udp套接字和 user_socket_pair 收发包，沟通用户态程序和对端的tcp通信者
 * 它在与对端tcp通信的时候，采取tcp协议的拥塞控制、窗口控制等算法
 */

// 哈希函数
static guint btcp_tcpconn_hash(gconstpointer key) {
    const struct btcp_tcpconn_handler *conn = (const struct btcp_tcpconn_handler *)key;
    //计算哈希值
    uint64_t v = 0;

    in_addr_t addr = inet_addr(conn->peer_ip);
    uint16_t port = conn->peer_port;
    
    memcpy(&v, &addr, sizeof(in_addr_t));
    memcpy( ((unsigned char*)&v)+sizeof(in_addr_t), &port, sizeof(port));
    return  v%4294967291; // 4294967291是一个素数
}

// 相等比较函数
static gboolean btcp_tcpconn_equal(gconstpointer a, gconstpointer b) {
    const struct btcp_tcpconn_handler *conn1 = (const struct btcp_tcpconn_handler *)a;
    const struct btcp_tcpconn_handler *conn2 = (const struct btcp_tcpconn_handler *)b;
    // 比较相等
    return conn1->peer_port == conn2->peer_port && 
        strncmp(conn1->peer_ip, conn2->peer_ip, INET_ADDRSTRLEN)==0;
}

// 释放键的函数
static void btcp_tcpconn_key_destroy(gpointer key) {

    struct btcp_tcpconn_handler* k = (struct btcp_tcpconn_handler*)key;
    
    g_info("Destroying key: ip=%s, port=%d", k->peer_ip, k->peer_port);
    btcp_destroy_tcpconn(k, true);
    free(key);  // 释放动态分配的结构体内存
}

// 释放值的函数
static void btcp_tcpconn_value_destroy(gpointer value) {
    // 注意：如果键和值是同一个指针，这里不需要释放内存， 也不需要做 btcp_destroy_tcpconn
    // 这死后都不应该访问value了，读都不行。这里是出过内存越界访问的bug的！
   struct btcp_tcpconn_handler* v = (struct btcp_tcpconn_handler*)value;
   //g_info("Destroying value: ip=%s, port=%d", v->peer_ip, v->peer_port);
    
}

//发送报文后，超时时间设置为多少
int btcp_get_timeout_sec(struct btcp_tcpconn_handler *handler)
{
    int v = handler->rtt.current_rtt_msec * 5 / 1000; // 5个rtt
    if (v <= 1 || v > 5)
    {
        v = 2; //设置为1不合适，因为当前定时器的精度是秒级别的，1的话可能很快就超时了，只要系统时间的秒发生了变化
    }
    return v;
}



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
        server_addr.sin_port = htons(port);

        // 绑定套接字到指定端口
        if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
        {
            btcp_errno = ERR_INIT_UDP_FAIL;
            close(sockfd);
            return -1;
        }

        srv->udp_socket = sockfd;
    }
    {
        pthread_mutex_init(&srv->all_connections_mutex, NULL);
        srv->all_connections = g_hash_table_new_full(btcp_tcpconn_hash, btcp_tcpconn_equal,
                btcp_tcpconn_key_destroy, btcp_tcpconn_value_destroy);
        if (srv->all_connections == NULL)
        {
            btcp_errno = ERR_MEM_ERROR;
            return -1;
        }
        
    }
    return 0;
}  
 
struct btcp_tcpconn_handler *  btcp_handle_sync_rcvd1(char * bigbuffer, 
            struct btcp_tcpsrv_handler* srv, const struct sockaddr_in * client_addr)
{
    struct btcp_tcpconn_handler * handler = NULL;

    union btcp_tcphdr_with_option *tcphdr = (union btcp_tcphdr_with_option *)bigbuffer;
    struct btcp_tcphdr * hdr = &tcphdr->base_hdr;

    btcp_print_tcphdr((const char *)hdr, "recv sync:");

    if (!btcp_check_tcphdr_flag(FLAG_SYN, (hdr->doff_res_flags)) ) 
    {
        btcp_errno = ERR_INVALID_PKG;
        return NULL;
    }
    if ( ntohs(hdr->dest) != srv->local_port)
    {
        btcp_errno = ERR_PORT_MISMATCH;
        return NULL;
    }
    
    
    {
        handler = malloc(sizeof(struct btcp_tcpconn_handler));
        if (handler == NULL)
        {
            btcp_errno = ERR_MEM_ERROR;
            return NULL;
        }
        if (btcp_init_tcpconn(handler))
        {
            free(handler);
            return NULL;
        }
        handler->peer_port = ntohs(hdr->source);
        char peer_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr->sin_addr, peer_ip, INET_ADDRSTRLEN);

     
        strcpy(handler->peer_ip, peer_ip);

        int mtu = btcp_get_route_mtu(peer_ip);
        if (mtu > 60)
        {
            handler->mss = mtu - 60;
        }
        else
        {
            btcp_errno = ERR_GET_MTU_FAIL;
            free(handler);
            return NULL;
        }
        //todo:测试用，要注释掉
        handler->mss = 5;
        
        handler->cong_wnd = 1;

       
        handler->local_port = srv->local_port;
        
        handler->local_seq = btcp_get_random() % UINT16_MAX;
        
        handler->status = SYNC_RCVD;
        handler->udp_socket = srv->udp_socket;
    }

    handler->peer_seq = btcp_sequence_step_forward(ntohl(hdr->seq), 1);
    btcp_recv_queue_set_start_seq(&handler->recv_buf, handler->peer_seq);
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
    //收到了对端的合法报文，认为该连接处于活跃状态，更新保活时间戳
    handler->alive_time_stamp = time(NULL);

    //send ack package
    memset(tcphdr, 0, sizeof(union btcp_tcphdr_with_option));
    hdr->ack_seq = htonl(handler->peer_seq);
    btcp_set_tcphdr_flag(FLAG_ACK, &(hdr->doff_res_flags));
    
    btcp_set_tcphdr_flag(FLAG_SYN, &(hdr->doff_res_flags));
 
    hdr->dest = htons(handler->peer_port);
    hdr->source = htons(handler->local_port);
    hdr->seq = htonl(handler->local_seq);
    int recv_wndsz = btcp_recv_queue_get_available_space(&handler->recv_buf);
    hdr->window = htons(recv_wndsz);
    
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
        free(handler);
        return NULL;
    }
    handler->status = SYNC_RCVD;

    btcp_print_tcphdr((const char *)hdr, "send ack:");
    return handler;
}

int btcp_handle_sync_rcvd2(char * bigbuffer,  struct btcp_tcpconn_handler * handler, 
                        const struct sockaddr_in * client_addr)
{
    
    
    union btcp_tcphdr_with_option *tcphdr = (union btcp_tcphdr_with_option *)bigbuffer;
    struct btcp_tcphdr * hdr = &tcphdr->base_hdr;

    btcp_print_tcphdr((const char *)hdr, "recv ack:");

    if ( !btcp_check_tcphdr_flag(FLAG_ACK, (hdr->doff_res_flags)) ) 
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
    handler->local_seq = btcp_sequence_step_forward(handler->local_seq, 1);
    btcp_send_queue_set_start_seq(&handler->send_buf, handler->local_seq);

    if ((handler->peer_seq) !=ntohl(hdr->seq))
    {
        
        btcp_errno = ERR_SEQ_MISMATCH;
        return -1;
    }
    

    // 创建一对已连接的 Unix Domain Socket
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, handler->user_socket_pair) == -1) {
        perror("socketpair");
        exit(EXIT_FAILURE);
    }
    g_info("create socketpair:%d,%d", handler->user_socket_pair[0], handler->user_socket_pair[1]);
    
    btcp_set_socket_nonblock(handler->user_socket_pair[0]);
    btcp_set_socket_nonblock(handler->user_socket_pair[1]);

    //收到了对端的合法报文，认为该连接处于活跃状态，更新保活时间戳
    handler->alive_time_stamp = time(NULL);
    
    handler->status = ESTABLISHED;
    printf("established!\n");
    return 0;
}

int btcp_increase_cong_wnd(struct btcp_tcpconn_handler *handler)
{
    if (handler->cong_wnd > 1024*1024) //很大很大了，就不加了
    {
        return 0;
    }

    if (handler->cong_wnd < handler->cong_wnd_threshold)
    {
        handler->cong_wnd++;
        g_info("congest wnd:%d", handler->cong_wnd);
    }
    else // 超过 阈值后，就是每个rtt轮次加1， 我这里实现为简单的每秒加1
    {
        
        time_t current = time(NULL);
        if (current == handler->cong_wnd_prev_inc_time)
        {
            //不加，这一秒时间间隔里已经加过了
        }
        else
        {
            handler->cong_wnd++;
            g_info("congest wnd:%d", handler->cong_wnd);
            handler->cong_wnd_prev_inc_time = current; //记录时间，避免重复加
        }
    }
    return 0;
}
int btcp_shrink_cong_wnd(struct btcp_tcpconn_handler *handler, bool quick)
{
    handler->cong_wnd_threshold = handler->cong_wnd / 2;
    if (handler->cong_wnd_threshold < 4)
    {
        handler->cong_wnd_threshold = 8;
    }

    if (!quick)
    {
        handler->cong_wnd = 1;
    }
    else
    {
        handler->cong_wnd = handler->cong_wnd_threshold;
    }

    g_info("congest wnd:%d", handler->cong_wnd);
    
    
    return 0;
}

int btcp_handle_ack(union btcp_tcphdr_with_option *tcphdr, struct btcp_tcpconn_handler *handler)
{
    struct btcp_tcphdr * hdr = &tcphdr->base_hdr;
    uint32_t ack_seq32 = ntohl(hdr->ack_seq);
    uint64_t ack_seq64 = ack_seq32;

    
    if (ack_seq32 < handler->local_seq) //可能发生了回绕
    {
        ack_seq64 = btcp_sequence_round_out(ack_seq32);

        #if 0

        const uint64_t DISTANCE = handler->mss * 1024;
        if ( handler->local_seq + DISTANCE > UINT32_MAX) // 当前local_seq很靠近UINT32_MAX
        {
            ack_seq64 = btcp_sequence_round_out(ack_seq32);
        }
        else
        {
            g_warning("ack seq invalid. throw away.");
            return 0;
        }
        #endif
        
    }
    //挥手时候的特殊情况
    if (handler->status == FIN_WAIT1)
    {
        struct btcp_range range;
        range.from = handler->local_seq;
        range.to = handler->local_seq;

        handler->status = FIN_WAIT2;
        handler->local_seq = ack_seq32;
        btcp_send_queue_set_start_seq(&handler->send_buf, handler->local_seq);

        g_info("enter fin_wait2 status");

        g_info("local sequence step forward to %u", handler->local_seq);
        // 删除可能的定时器
        btcp_timer_remove_range(&handler->timeout, &range);
        return 0;
    }
    if (handler->status == LAST_ACK)
    {
        struct btcp_range range;
        range.from = handler->local_seq;
        range.to = handler->local_seq;

        handler->status = CLOSED;

        handler->local_seq = ack_seq32;
        btcp_send_queue_set_start_seq(&handler->send_buf, handler->local_seq);

        g_info("enter closed status");

        // 删除可能的定时器
        btcp_timer_remove_range(&handler->timeout, &range);
        return 0;
    }

    //后面是正常数据的ack处理
    btcp_rtt_update_rtt(&handler->rtt, ack_seq32);
    
    if (ack_seq64 > ((uint64_t)handler->local_seq + (uint64_t)handler->send_buf.capacity)) // 大太多了，就算是累计确认也不能差这么多
    {
        g_warning("ack sequence is too big! %u, %u", ack_seq32, handler->local_seq);
        return -1;
    }
    if (handler->local_seq == ack_seq32) // 收到对当前sequence的重复确认，非正常情况
    {
        handler->repeat_ack++;
        if (handler->repeat_ack >= 3) //连续收到3次或者以上，触发窗口缩小和重发
        {
            handler->repeat_ack = 0;
            // 修改发送窗口大小
            btcp_shrink_cong_wnd(handler, true);
            btcp_timer_remove_by_from(&handler->timeout, handler->local_seq);//删除计时器里起始seq等于local_seq的记录
            //btcp_try_send(handler); //删掉计时器里的记录，其实后面就会比较及时的重发

            g_info("repeated ack 3 times!");
        }
    }
    else
    {
        handler->repeat_ack = 0;
        // todo:这里真的就信任网络上发过来的一个ack报文而直接修改seq吗？
        // 是不是应该做一些检查和限制
        struct btcp_range range;
        range.from = handler->local_seq;
        range.to = ack_seq64-1; //闭区间，所以要减一， 这是一个bug被修复了

        handler->local_seq = ack_seq32;
        btcp_send_queue_set_start_seq(&handler->send_buf, ack_seq64);
        g_info("local sequence step forward to %u", handler->local_seq);
        // 删除可能的定时器
        btcp_timer_remove_range(&handler->timeout, &range);

        btcp_increase_cong_wnd(handler);
    }
    //继续处理selective ack的信息
    {
        int hdr_len = btcp_get_tcphdr_offset(&hdr->doff_res_flags);
        if (hdr_len > sizeof(struct btcp_tcphdr))
        {
            int offset = sizeof(struct btcp_tcphdr);
            uint8_t opt_kind = *(uint8_t*)(tcphdr->options+offset);
            offset++;
            uint8_t opt_len = *(uint8_t*)(tcphdr->options+offset);
            offset++;
            if (opt_kind == 5 && opt_len == 10)
            {
                struct btcp_range opt_range;
                opt_range.from = ntohl(*(uint32_t*)(tcphdr->options+offset));
                offset += 4;
                opt_range.to   = ntohl(*(uint32_t*)(tcphdr->options+offset));
                offset += 4;

                btcp_sack_blocklist_add_record(&handler->sack, &opt_range);
                g_info("get a sack block [%llu, %llu]", opt_range.from, opt_range.to);
            }
        }
    }

    

    return 0;
}



 int btcp_keep_alive(struct btcp_tcpconn_handler *handler, char *bigbuffer, bool is_server)
{
    time_t current = time(NULL);
    if (is_server)
    {
        if (handler->status == SYNC_RCVD || handler->status == SYNC_SENT)
        {
            if (current - 30 >  handler->alive_time_stamp)
            {
                return 1; // 已经超时了，需要上层处理
            }
            else
            {
                return 0;
            }
        }
    }
    // 对处于ESTABLISHED状态的conn进行检查，如果超过15s，就发保活请求包，如果超过60s，就认为超时了
    if (handler->status == ESTABLISHED)
    {
        if (current - 60 > handler->alive_time_stamp)
        {
            return 1;
        }
        if (current - 15 > handler->alive_time_stamp &&
            current - 15 > handler->keepalive_request_time)
        {
            #if 1 
            union btcp_tcphdr_with_option *tcphdr = (union btcp_tcphdr_with_option *)bigbuffer;
            struct btcp_tcphdr * hdr = &tcphdr->base_hdr;

            memset(tcphdr, 0, sizeof(union btcp_tcphdr_with_option));
            hdr->ack_seq = htonl(handler->peer_seq);
            btcp_set_tcphdr_flag(FLAG_ACK, &(hdr->doff_res_flags));
            hdr->dest = htons(handler->peer_port);
            hdr->source = htons(handler->local_port);
            hdr->seq = htonl(btcp_sequence_step_back(handler->local_seq, 1));
            int recv_wndsz = btcp_recv_queue_get_available_space(&handler->recv_buf);
            hdr->window = htons(recv_wndsz); 

            int offset = sizeof(struct btcp_tcphdr);
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
            g_info("send keep alive request, %d", __LINE__);
            handler->keepalive_request_time = current;
            
            #endif
        }
    }
    

    return 0;
}

int btcp_throw_data_to_user(struct btcp_tcpconn_handler * handler)
{
    int size = btcp_recv_queue_size(&handler->recv_buf);
    if (size <= 0)
    {
        g_warning("unexpected data size! %d, %s %d", size, __FILE__, __LINE__);
        return 0;
    }
    unsigned char buf[1024];
    while (size > 0)
    {
        int len = sizeof(buf);
        if (len > size)
        {
            len = size;
        }
        btcp_recv_queue_dequeue(&handler->recv_buf, buf, len);
        size -= len;
        int written = send(handler->user_socket_pair[1], buf, len, MSG_NOSIGNAL);
        if (written!= len)
        {
            g_warning("write user socket pair failed!%d!=%d, (%s, %d)", written, len, __FILE__, __LINE__);
        }
        else
        {
            g_info("thow %d bytes data to users", written);
        }
        
    }
    return 0;
}
#if 0
static btcp_check_sequence(struct btcp_tcpconn_handler * handler, uint32_t seq32, bool is_keepavlie_req)
{
    if (seq32 < handler->peer_seq) //处理发过来的报文seq偏小的特殊情况 
    {
        if (is_keepavlie_req) //前面判断过这是个keep alive报文，那没事，是合法的
        {

        }
        else 
        {
            //还可能发生了seq回绕, 超出UINT32_MAX又从0开始了
            // 特征就是当前记录的sequence比较靠近UINT32_MAX
            const uint64_t DISTANCE = handler->mss * 1024;
            if ( handler->peer_seq + DISTANCE > UINT32_MAX) 
            {
                int recv_wndsz = btcp_recv_queue_get_available_space(&handler->recv_buf);
                if (btcp_sequence_round_out(seq32) - handler->peer_seq > recv_wndsz) // 那也太大了
                {
                    g_warning("invalid pkg at %s %d, %u,%u", __FILE__, __LINE__, seq32, handler->peer_seq);
                    return 0;
                }
            }
            else //不属于回绕的情况
            {
                g_warning("invalid pkg at %s %d, %u,%u", __FILE__, __LINE__, seq32, handler->peer_seq);
                return 0;
            }
        }
    } 
    if (seq32 > handler->peer_seq)
    {
        // 有一种情况是 peer_seq刚刚回绕到 小整数例如2，但seq32还是旧一点的seq例如40亿
        // 这种就是过期的请求
        if (step > 1024 * 1024 * 100 &&
            queue->start_seq < 1024 * 100)
        {
            step = btcp_sequence_round_out(queue->start_seq) - position;
        }
    }
}
#endif


int btcp_enter_close_wait(struct btcp_tcpconn_handler *handler, 
                        union btcp_tcphdr_with_option *tcphdr, 
                        char *bigbuffer)
{
    int iret;
    if ( (iret = btcp_send_fin_response(handler, tcphdr, bigbuffer)) == 0)
    {
        handler->status = CLOSE_WAIT;
        return 0;
    }
    g_info("enter close_wait status");
    return -1;
}
int btcp_enter_time_wait(struct btcp_tcpconn_handler *handler, 
                        union btcp_tcphdr_with_option *tcphdr, 
                        char *bigbuffer)
{
    int iret;
    if ( (iret = btcp_send_fin_response(handler, tcphdr, bigbuffer)) == 0)
    {
        handler->status = TIME_WAIT;
        return 0;
    }
    g_info("enter time_wait status");
    return -1;
}
int btcp_send_fin_response(struct btcp_tcpconn_handler *handler, 
                        union btcp_tcphdr_with_option *tcphdr, 
                        char *bigbuffer)
{
    memset(tcphdr, 0, sizeof(union btcp_tcphdr_with_option));
    struct btcp_tcphdr * hdr = &tcphdr->base_hdr;
    hdr->ack_seq = htonl(handler->peer_seq);
    btcp_set_tcphdr_flag(FLAG_ACK, &(hdr->doff_res_flags));
    hdr->dest = htons(handler->peer_port);
    hdr->source = htons(handler->local_port);
    hdr->seq = htonl(handler->local_seq);
    int recv_wndsz = btcp_recv_queue_get_available_space(&handler->recv_buf);
    hdr->window = htons(recv_wndsz);
    int offset = sizeof(struct btcp_tcphdr);
    
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
    btcp_print_tcphdr(bigbuffer, "send fin response package:");
    return 0;
}

int btcp_handle_data_rcvd(char * bigbuffer, int pkg_len, struct btcp_tcpconn_handler * handler, 
            const struct sockaddr_in * client_addr)
{
    
    union btcp_tcphdr_with_option *tcphdr = (union btcp_tcphdr_with_option *)bigbuffer;
    struct btcp_tcphdr * hdr = &tcphdr->base_hdr;

    btcp_print_tcphdr((const char *)hdr, "recv tcp:");

    if ( ntohs(hdr->source) != handler->peer_port ||
        ntohs(hdr->dest) != handler->local_port)
    {
        btcp_errno = ERR_PORT_MISMATCH;
        return -1;
    }
    
    int data_len = pkg_len - btcp_get_tcphdr_offset(&hdr->doff_res_flags);
    if (data_len < 0)
    {
        btcp_errno = ERR_INVALID_PKG;
        return -1;
    }
    uint32_t ack_seq32 = ntohl(hdr->ack_seq);
    uint32_t seq32 = ntohl(hdr->seq); 

    

    int offset = btcp_get_tcphdr_offset(&hdr->doff_res_flags);
    
    bool got_keepalive_request = false;
    if ( btcp_check_tcphdr_flag(FLAG_ACK, (hdr->doff_res_flags)) ) // 如果带有ack标记
    {
        if (seq32 == btcp_sequence_step_back(handler->peer_seq, 1) &&
            ack_seq32 == handler->local_seq &&
            data_len == 0) // is a keepalive request，也需要ack
        {
            g_info("got a keep alive request");
            got_keepalive_request = true;
            
        }
        else
        {
            //普通ack的处理，可能需要移动发送窗口
            // 如果没有带用户数据，就不需要再ack
            btcp_handle_ack(tcphdr, handler); 
        }
        
    }
    //收到报文不在接收窗口内，又不是keep alive报文，那就直接丢弃。
    // 这类非法报文，可能是在途超时到达的老报文或者恶意者攻击
    /*
    TCP 拥塞控制中的一个经典问题，称为 “重传冗余” 或 “不必要的重传”。当发送窗口较大且有一定丢包率时
    例如：假设发送窗口大小为3，起始seq=100，发送了 p1 p2 p3 三个包，其中p1途中丢失了，对端收到了p2 p3 ，
    会发送2个ack，ack_seq都会等于100。发送端一段时间后就会发现p1, p2 p3都没有被ack而超时，这时候又
    重新发送p1 p2 p3。对端收到p1后立即移动接收窗口获得了完整的p1 p2 p3, sequence移动到p3报文的末尾字节
    位置，再接着收到重发的p2 p3，这时候的p2 p3是白白重复传输的。

    优化方案可以使用SACK选项，即选择性ack，对端收到p2, p3后ack的时候，在首部选项字段里也带上对p2 p3的ack
    发送端收到sack信息后，不再重复发送
    */
    int recv_wndsz = btcp_recv_queue_get_available_space(&handler->recv_buf);
    if (seq32 < handler->peer_seq &&  
        btcp_sequence_round_out(seq32) -handler->peer_seq > recv_wndsz)
    {
        if (!got_keepalive_request)
        {
            g_warning("invalid pkg at %s %d, %u,%u", __FILE__, __LINE__, seq32, handler->peer_seq);
            return 0;
        }
        
    } 
    unsigned char sack_option[10] = {0}; 
    handler->peer_recv_wnd_sz = ntohs(hdr->window); 
    if (data_len > 0)
    {
        uint64_t from_seq = ntohl(hdr->seq);
        uint64_t to_seq = from_seq + data_len - 1;
        // save data to recv queue
        if (btcp_recv_queue_save_data(&handler->recv_buf, from_seq, to_seq,
                                      bigbuffer + offset))
        {
            /*
            // 还是不要返回失败，免得被攻击者利用
            btcp_errno = ERR_SEQ_MISMATCH;
            return -1;
            */
            g_warning("invalid pkg at %s %d", __FILE__, __LINE__);
            return 0;
        }
        g_info("data_len:%d, peer_seq:%u", data_len, handler->peer_seq);
        
        if ((handler->peer_seq ) == ntohl(hdr->seq)) // 收到了想要的下一个（顺序）报文，需要移动接收窗口
        {
            //移动的大小不一定就等于data_len，因为可能之前已经收到过 后发先至 的数据段，与这个报文连成一片。
            int steps = btcp_recv_queue_try_move_wnd(&handler->recv_buf);
            if (steps < 0)
            {
                g_warning("btcp_recv_queue_try_move_wnd() failed! %d", steps);
                btcp_errno = ERR_SEQ_MISMATCH;
                return -1;
            }
            g_info("recv wnd move %d bytes", steps);

            // sequence step forward
            handler->peer_seq = btcp_sequence_step_forward(handler->peer_seq, steps);
            //本来这里也需要同步的修改recv queue里的expected_seq，但try_move_wnd函数里面已经修改了

            g_info("peer_seq changes to:%u, start_seq:%u, tail:%d",  handler->peer_seq,
                    handler->recv_buf.start_seq,
                    handler->recv_buf.tail);
            //向应用层抛数据
            btcp_throw_data_to_user(handler);
             
        }
        else
        {
            //准备一段selective ack option数据，追加到tcp首部里
            /*
            格式：
            Kind: 5 one byte
            Length: 可变（取决于 SACK 块的数量） on byte。
            Value: 一个或多个 SACK 块，每个 SACK 块包含两个 32 位的序列号：
                Left Edge：已接收数据块的起始序列号。
                Right Edge：已接收数据块的结束序列号（即下一个期望的序列号）。
            */
            *(uint8_t*)(sack_option + 0) = 5; 
            *(uint8_t*)(sack_option + 1) = 10;

            uint32_t left = seq32;
            *(uint32_t*)(sack_option + 2) = htonl(seq32);

            uint32_t right = btcp_sequence_round_in((uint64_t)seq32+data_len-1);
            *(uint32_t*)(sack_option + 6) = htonl(right);

            g_info("send a sack block[%u, %u]", left, right);
        }
    }
    //如果收到了数据（就算不是顺序的），或者收到keepalive请求
    //都需要发送ack 应答
    if (data_len > 0 || got_keepalive_request ) 
    {
    
        memset(tcphdr, 0, sizeof(union btcp_tcphdr_with_option));
        hdr->ack_seq = htonl(handler->peer_seq);
        btcp_set_tcphdr_flag(FLAG_ACK, &(hdr->doff_res_flags));
        hdr->dest = htons(handler->peer_port);
        hdr->source = htons(handler->local_port);
        hdr->seq = htonl(handler->local_seq);
        int recv_wndsz = btcp_recv_queue_get_available_space(&handler->recv_buf);
        hdr->window = htons(recv_wndsz);
        int offset = sizeof(struct btcp_tcphdr);
        if (sack_option[0] != 0)
        {
            memcpy(tcphdr->options+offset, sack_option, sizeof(sack_option));
            offset += sizeof(sack_option);

            *(uint16_t*)(tcphdr->options+offset) = 0; // padding 首部的长度要求是4B的倍数
            offset += 2; 
        }
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
        btcp_print_tcphdr(bigbuffer, "send ack package:");
    }

    //收到了对端的合法报文，认为该连接处于活跃状态，更新保活时间戳
    handler->alive_time_stamp = time(NULL);
    g_info("modify alive timestamp to current");

// 如果带有fin标记, 是挥手请求
    if (btcp_check_tcphdr_flag(FLAG_FIN, (hdr->doff_res_flags)) ) 
    {
        //对端发带FIN的报文过来，有两种情况
        if (handler->status == ESTABLISHED || handler->status == FIN_WAIT2)
        {
            btcp_recv_queue_save_fin_req(&handler->recv_buf, seq32);
            g_info("rcvd fin request, seq=%u, save to recv queue", seq32);
        }
        
    }

    // 每次收到对端的报文，都看看是不是该处理对端发的fin request了
    // 这样能够及时处理fin请求：
    // 1）要么数据前面都处理完了，这次报文就是一个FIN 请求，所以这次就立即处理了
    // 2）要么FIN 请求相比数据是后发先至的，在recv_buf里躺着，每次来一个数据报文，都可能让recv_buf里seq
    //    连成一片，这时候上抛完用户数据，就可以处理fin 请求了
    if (handler->recv_buf.fin_seq >= 0 && handler->recv_buf.fin_seq == handler->peer_seq)
    {
        g_info("fin request need to be processed");
        // fin请求占用 1 sequence
        handler->peer_seq = btcp_sequence_step_forward(handler->peer_seq, 1);
        btcp_recv_queue_set_start_seq(&handler->recv_buf, handler->peer_seq);
        shutdown(handler->user_socket_pair[1], SHUT_WR);
        // 有两种情况
        int iret;
        if (handler->status == ESTABLISHED)
        {
            iret = btcp_enter_close_wait(handler, tcphdr, bigbuffer);
        }
        else if (handler->status == FIN_WAIT2)
        {
            iret = btcp_enter_time_wait(handler, tcphdr, bigbuffer);
        }

        
    }
    return 0;
}
// 创建背后通信用的udp套接字
static int btcp_tcpcli_init_udp(struct btcp_tcpconn_handler * handler) 
{
    if (handler == NULL) { btcp_errno = ERR_INVALID_ARG; return -1;}
    
    

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
    g_info("local port:%d\n", handler->local_port );
    return 0;
}


int btcp_tcpcli_connect(const char * ip, short int port, struct btcp_tcpconn_handler * handler)
{
    if (strlen(ip) >= INET_ADDRSTRLEN) {btcp_errno = ERR_INVALID_ARG; return -1;}
    if (btcp_init_tcpconn(handler))
    {
        return -1;
    }
    strcpy(handler->peer_ip, ip);
    g_info("peer ip:%s", handler->peer_ip);
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
    //todo:测试用，要注释掉
    handler->mss = 5;
   
    handler->cong_wnd = 1;
    handler->cong_wnd_threshold = 8;
  
    
    if (btcp_tcpcli_init_udp(handler)) { return -1;}
    
    g_info("in connect(), peer ip:%s, mss:%d, peer_port:%d\n", handler->peer_ip, handler->mss, handler->peer_port);
    // three handshakes
    {
        union btcp_tcphdr_with_option tcphdr;
        struct btcp_tcphdr * hdr = &tcphdr.base_hdr;
        memset(&tcphdr, 0, sizeof(union btcp_tcphdr_with_option));
        hdr->dest = htons(port);
        
        hdr->source = htons(handler->local_port);
        handler->peer_port = port;
        handler->local_seq = btcp_get_random() % UINT16_MAX;
        
        int recv_wndsz = btcp_recv_queue_get_available_space(&handler->recv_buf);
        hdr->window = htons(recv_wndsz);
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
    g_info("in connect(), peer ip:%s, mss:%d, peer_port:%d\n", handler->peer_ip, handler->mss, handler->peer_port);
  
    return 0;
}
int btcp_handle_sync_sent(char * bigbuffer,  struct btcp_tcpconn_handler * handler)
{
    if (handler->status != SYNC_SENT)
    {
        return -1;
    }
    g_info("peer ip:%s, mss:%d, peer_port:%d\n", handler->peer_ip, handler->mss, handler->peer_port);
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
    
    handler->peer_seq = btcp_sequence_step_forward(ntohl(hdr->seq), 1);
    btcp_recv_queue_set_start_seq(&handler->recv_buf, handler->peer_seq);
    uint32_t ack_seq = ntohl(hdr->ack_seq);
    if (ack_seq != (handler->local_seq + 1) )
    {
        btcp_errno = ERR_SEQ_MISMATCH;
        return -1;
    }
    handler->peer_recv_wnd_sz = ntohs(hdr->window);


    handler->local_seq = btcp_sequence_step_forward(handler->local_seq, 1);
    btcp_send_queue_set_start_seq(&handler->send_buf, handler->local_seq);
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

    //收到了对端的合法报文，认为该连接处于活跃状态，更新保活时间戳
    handler->alive_time_stamp = time(NULL);

    //send ack package
    memset(tcphdr, 0, sizeof(union btcp_tcphdr_with_option));
    hdr->ack_seq = htonl(handler->peer_seq);
    btcp_set_tcphdr_flag(FLAG_ACK, &(hdr->doff_res_flags));

    hdr->dest = htons(handler->peer_port);
    hdr->source = htons(handler->local_port);
    hdr->seq = htonl(handler->local_seq);
    int recv_wndsz = btcp_recv_queue_get_available_space(&handler->recv_buf);
    hdr->window = htons(recv_wndsz);
    
    offset = sizeof(struct btcp_tcphdr);
    btcp_set_tcphdr_offset(offset, &hdr->doff_res_flags);

    struct sockaddr_in server_addr;
    // 初始化服务器地址结构
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(handler->peer_ip);
    server_addr.sin_port = htons(handler->peer_port);

    g_info("send udp package to %s, %d, len:%d\n", handler->peer_ip, handler->peer_port, offset);
    

    int iret = sendto(handler->udp_socket, hdr, offset, 0, (const struct sockaddr *)&server_addr, sizeof(server_addr));
    if (iret != offset)
    {
        perror("sendto:");
        printf("iret=%d\n", iret);
        btcp_errno = ERR_UDP_COMMM_FAIL;
        close(handler->udp_socket);
        return -1;
    }
    handler->status = ESTABLISHED;

    // 创建一对已连接的 Unix Domain Socket
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, handler->user_socket_pair) == -1) {
        perror("socketpair");
        exit(EXIT_FAILURE);
    }
    g_info("socketpair() success, %d, %d", handler->user_socket_pair[0],
                                            handler->user_socket_pair[1]);
    btcp_set_socket_nonblock(handler->user_socket_pair[0]);
    btcp_set_socket_nonblock(handler->user_socket_pair[1]);

    btcp_print_tcphdr((const char *)hdr, "send ack:");
    printf("established!\n");
    return 0;
}
int btcp_enter_fin_wait1(struct btcp_tcpconn_handler *handler, 
                    char *bigbuffer)
{
    int iret;
    if ( (iret = btcp_send_fin_request(handler, bigbuffer,  NULL)) == 0)
    {
        handler->status = FIN_WAIT1;
    }
    g_info("enter fin_wait1 status");
    return iret;
}
int btcp_enter_last_ack(struct btcp_tcpconn_handler *handler, 
                    char *bigbuffer)
{
    uint32_t ack_seq = handler->peer_seq;
    int iret;
    if ( (iret = btcp_send_fin_request(handler, bigbuffer,  &ack_seq)) == 0)
    {
        handler->status = LAST_ACK;
    }
    g_info("enter last_ack status");
    return iret;
}

int btcp_send_fin_request(struct btcp_tcpconn_handler *handler, 
                    char *bigbuffer, 
                    uint32_t * ack_seq)
{
    struct btcp_range c_range; 
    c_range.from = handler->send_buf.fin_seq;
    c_range.to = handler->send_buf.fin_seq;
    if (btcp_timer_find_event(&handler->timeout, &c_range, 
                        sizeof(struct btcp_range), btcp_range_equal) != NULL)
    {
        //还有未超时的已发送的在途的fin请求
        return 0;
    }
    g_info("fin request to send, fin seq: %llu", handler->send_buf.fin_seq);

    struct btcp_tcphdr *hdr = (struct btcp_tcphdr *)bigbuffer;
    memset(hdr, 0, sizeof(struct btcp_tcphdr));
    hdr->dest = htons(handler->peer_port);
    hdr->source = htons(handler->local_port);
    int recv_wndsz = btcp_recv_queue_get_available_space(&handler->recv_buf);
    hdr->window = htons(recv_wndsz);
    hdr->seq = htonl(handler->send_buf.fin_seq);
    int offset = sizeof(struct btcp_tcphdr);
    if (ack_seq != NULL)
    {
        hdr->ack_seq = htonl(*ack_seq);
        btcp_set_tcphdr_flag(FLAG_ACK, &hdr->doff_res_flags);
    }
    btcp_set_tcphdr_flag(FLAG_FIN, &hdr->doff_res_flags);
    btcp_set_tcphdr_offset(offset, &hdr->doff_res_flags);

    struct sockaddr_in server_addr;

    // 初始化服务器地址结构
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(handler->peer_ip);
    server_addr.sin_port = htons(handler->peer_port);

    int sent_len = sendto(handler->udp_socket, hdr, offset, 0, (const struct sockaddr *)&server_addr, sizeof(struct sockaddr_in));
    if (sent_len < 0) // udp发包，不存在只发部分报文的情况，要么完整报文，要么负1
    {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
        {
            g_warning("udp socket sendto failed, EAGAIN");
            return 0;
        }
        btcp_errno = ERR_UDP_COMMM_FAIL;
        close(handler->udp_socket);
        return  -1; 
    }
    btcp_print_tcphdr(bigbuffer, "send fin request:");
    //  记录超时事件, timer里记录的range的sequence都是32bit范围内的值，方便与ack报文的sequence对应
    int sec = btcp_get_timeout_sec(handler);
    if (btcp_timer_add_event(&handler->timeout, sec, &c_range, sizeof(struct btcp_range),
                             btcp_range_equal))
    {
        g_warning("btcp_timer_add_event() failed!\n");
    }
    else
    {
        g_info("add fin request to timer:[%llu, %llu]", c_range.from, c_range.to);
    }
    return 0;
}

int btcp_try_send(struct btcp_tcpconn_handler *handler)
{
    int retcode = -1;
    //////////////////////////////////////////////
    //准备要发送的数据
    //////////////////////////////////////////////
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
    #if 0
    g_info("send_wndsz:%d, mss:%d, cong_wnd:%d, peer wndsz:%d", 
            send_wndsz,
            handler->mss,
            handler->cong_wnd,
            handler->peer_recv_wnd_sz);
      #endif      
    int datasz_in_queue = btcp_send_queue_size(&handler->send_buf);
    static unsigned char bigbuffer[100 * 1024] __attribute__((aligned(8))); // 临时用一下，不会跨线程，也不会超出函数的作用域
    struct btcp_range *range_to_send = NULL;
    GList *range_list_to_send = NULL;
    GList *range_list_sent = NULL;
    GList *range_list_result1 = NULL, *range_list_result2 = NULL, *combined_list = NULL;
    if (datasz_in_queue >0) // 队列里有数据要发送
    {
        // g_info("尝试发送数据，窗口大小为%d bytes, 发送缓冲里的数据有 %d bytes\n", send_wndsz, datasz_in_queue);
        // 发送窗口的范围，与已经发送的待ack报文覆盖的范围比较，找出需要发送的数据段 ，
        // 这里参与运算的seq/from/to使用uint64_t类型，且保证to >= from，即to可能大于UINT32_MAX
        range_to_send = malloc(sizeof(struct btcp_range));
        range_to_send->from = handler->send_buf.start_seq;
        if (send_wndsz <= datasz_in_queue)
        {
            range_to_send->to = (uint64_t)(handler->send_buf.start_seq) + send_wndsz - 1; // 闭区间，所以要减一
        }
        else
        {
            range_to_send->to = (uint64_t)(handler->send_buf.start_seq) + datasz_in_queue - 1; // 闭区间，所以要减一
        }
        //g_info("data range to send:[%llu, %llu]", range_to_send->from, range_to_send->to);

        
        range_list_to_send = g_list_append(NULL, range_to_send);

        
        if (btcp_timer_get_all_event(&handler->timeout, &range_list_sent) != 0)
        {
            btcp_errno = ERR_MEM_ERROR;
            goto btcp_try_send_out;
        }

        // 做一个特殊处理:如果是 range_to_send 的起止seq比较接近UINT32_MAX，就把rang_list_sent的起止
        // seq都做 btcp_seuquence_round_out，否则substract没有效果，导致重发一些报文
        // 典型的例子就是：发送窗口=[42亿， 100]，100是回绕了。而range_list_sent记录有[3,8]
        // 直接减的话没有效果，不会剔除掉[3,8]
        const uint64_t DISTANCE = handler->mss * 1024;
        if (range_to_send->from + DISTANCE > UINT32_MAX) // 当前local_seq很靠近UINT32_MAX
        {
            GList *iter = NULL;
            for (iter = range_list_sent; iter != NULL; iter = iter->next)
            {
                struct btcp_range *one_range = (struct btcp_range *)iter->data;
                if (one_range == NULL)
                {
                    continue;
                }
                if (one_range->from < DISTANCE) // 定时器里的记录range的seq很靠近0
                {
                    one_range->from = btcp_sequence_round_out(one_range->from);
                }
                if (one_range->to < DISTANCE)
                {
                    one_range->to = btcp_sequence_round_out(one_range->to);
                }
            }
            // sack_blocklist也是一样的处理一下
            for (iter = handler->sack.blocklist; iter != NULL; iter = iter->next)
            {
                struct btcp_range *one_range = (struct btcp_range *)iter->data;
                if (one_range == NULL)
                {
                    continue;
                }
                if (one_range->from < DISTANCE) // 定时器里的记录range的seq很靠近0
                {
                    one_range->from = btcp_sequence_round_out(one_range->from);
                }
                if (one_range->to < DISTANCE)
                {
                    one_range->to = btcp_sequence_round_out(one_range->to);
                }
            }
        }
        {

#ifdef _DETAIL_LOG_
            g_info("%lu, onraod data range:", range_list_sent);

            for (const GList *iter = range_list_sent; iter != NULL; iter = iter->next)
            {
                struct btcp_range *a_range = (struct btcp_range *)iter->data;
                g_info("[%llu, %llu]", a_range->from, a_range->to);
            }
#endif
        }
        
        if (btcp_range_subtract(range_list_to_send, range_list_sent, &range_list_result1))
        {
            btcp_errno = ERR_MEM_ERROR;
            goto btcp_try_send_out;
        }
        if (btcp_range_subtract(range_list_result1, handler->sack.blocklist, &range_list_result2))
        {
            btcp_errno = ERR_MEM_ERROR;
            goto btcp_try_send_out;
        }

        btcp_range_list_combine(range_list_result2, &combined_list);

        if (combined_list != NULL)
        {
            g_info("data to send:");
            for (const GList *iter = combined_list; iter != NULL; iter = iter->next)
            {
                struct btcp_range *a_range = (struct btcp_range *)iter->data;
                g_info("[%llu, %llu]", a_range->from, a_range->to);
            }
        }
        ///////////////////////////////////////////////////////
        // 发送数据，并插入超时等待队列
        ///////////////////////////////////////////////////////
        struct sockaddr_in server_addr;
        
        // 初始化服务器地址结构
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = inet_addr(handler->peer_ip);
        server_addr.sin_port = htons(handler->peer_port);
        for (const GList *iter = combined_list; iter != NULL; iter = iter->next)
        {
            struct btcp_range *a_range = (struct btcp_range *)iter->data;
            struct btcp_range b_range; // b_range的from to是 64bit 长范围的
            b_range.from = a_range->from;
            b_range.to = a_range->to;

            while (b_range.from <= b_range.to) // 如果超过mss，需要发送多次
            {
                int datalen = b_range.to - b_range.from + 1;
                if (datalen > handler->mss)
                {
                    datalen = handler->mss;
                }
                if (btcp_send_queue_fetch_data(&handler->send_buf, b_range.from, b_range.from + datalen - 1, bigbuffer + sizeof(struct btcp_tcphdr)))
                {
                    g_warning("!!!btcp_send_queue_fetch_data() failed\n");
                    break;
                }
                g_info("send a tcp package[%llu, %llu]", b_range.from, b_range.from + datalen - 1);

                struct btcp_tcphdr *hdr = (struct btcp_tcphdr *)bigbuffer;
                memset(hdr, 0, sizeof(struct btcp_tcphdr));
                hdr->dest = htons(handler->peer_port);
                hdr->source = htons(handler->local_port);
                int recv_wndsz = btcp_recv_queue_get_available_space(&handler->recv_buf);
                hdr->window = htons(recv_wndsz);
                hdr->seq = htonl(btcp_sequence_round_in(b_range.from));
                int offset = sizeof(struct btcp_tcphdr);
                btcp_set_tcphdr_offset(offset, &hdr->doff_res_flags);
                offset += datalen;

                // 模拟20%的丢包率 . todo:要改回去
                unsigned int r = btcp_get_random() % 20;
                int sent_len;
                if (r != 0)
                // if (1)
                {
                    sent_len = sendto(handler->udp_socket, hdr, offset, 0, (const struct sockaddr *)&server_addr, sizeof(server_addr));
                    if (sent_len < 0) // udp发包，不存在只发部分报文的情况，要么完整报文，要么负1
                    {
                        if (errno == EAGAIN || errno == EWOULDBLOCK)
                        {
                            retcode = 0;
                            g_warning("udp socket sendto failed, EAGAIN");
                            goto btcp_try_send_out;
                        }

                        btcp_errno = ERR_UDP_COMMM_FAIL;
                        close(handler->udp_socket);
                        goto btcp_try_send_out;
                    }
                    btcp_print_tcphdr(bigbuffer, "send:");
                }
                else
                {
                    g_info("package lost![%llu, %llu]", b_range.from, b_range.from + datalen - 1);
                    sent_len = offset;
                }
                // g_info("sent successfully, len:%d\n", sent_len);
                //  记录超时事件, timer里记录的range的sequence都是32bit范围内的值，方便与ack报文的sequence对应
                struct btcp_range c_range; // c_range是32bit短范围内的
                c_range.from = btcp_sequence_round_in(b_range.from);
                c_range.to = btcp_sequence_round_in(b_range.from + datalen - 1);

                int sec = btcp_get_timeout_sec(handler);
                if (btcp_timer_add_event(&handler->timeout, sec, &c_range, sizeof(struct btcp_range),
                                         btcp_range_equal))
                {
                    g_warning("btcp_timer_add_event() failed!\n");
                    break;
                }
                g_info("add to timer:[%llu, %llu]", c_range.from, c_range.to);

                if (c_range.from == handler->local_seq)
                {
                    uint32_t expected_ack_seq = btcp_sequence_round_in(c_range.to + 1);
                    btcp_rtt_add_send_record(&handler->rtt, expected_ack_seq);
                    g_info("add rec to rtt, ack seq:%u", expected_ack_seq);
                }

                b_range.from += datalen;
            }
        }
    }
    /////////////////////////////////////////////////
    //如果有fin请求要发送， 就发送fin请求
    ////////////////////////////////////////////////
    if (handler->send_buf.fin_seq >= 0 && handler->local_seq == handler->send_buf.fin_seq)
    {
        
        #if 1
        if (handler->status == ESTABLISHED || handler->status == FIN_WAIT1) //后面这个或，是因为可能需要重发fin req
        {
            btcp_enter_fin_wait1(handler, bigbuffer);
        }
        else if (handler->status == CLOSE_WAIT|| handler->status == LAST_ACK)//后面这个或，是因为可能需要重发fin req
        {
            btcp_enter_last_ack(handler, bigbuffer);
        }
        #endif
    }

    retcode = 0;
btcp_try_send_out:
    btcp_range_free_list(range_list_to_send);
    btcp_range_free_list(range_list_sent);
    btcp_range_free_list(range_list_result1);
    btcp_range_free_list(range_list_result2);
    btcp_range_free_list(combined_list);

    return retcode;
}

int btcp_check_send_timeout(struct btcp_tcpconn_handler *handler)
{
    struct btcp_range e;
    int len = sizeof(struct btcp_range);
    int timeout_occur = 0;
    while ( btcp_timer_check(&handler->timeout, &e, &len) == 1)
    {
        g_info("ack timeout, [%llu, %llu]\n", e.from, e.to);
        timeout_occur = 1; //有超时发生
        len = sizeof(struct btcp_range);
    }
    if (timeout_occur)
    {
        // 修改发送窗口大小
        btcp_shrink_cong_wnd(handler, false);
        
    }
    return timeout_occur;
}
int btcp_init_tcpconn(struct btcp_tcpconn_handler *handler)
{
    memset(handler, 0, sizeof(struct btcp_tcpconn_handler));
    
    if (!btcp_recv_queue_init(&handler->recv_buf, DEF_RECV_BUFSZ))
    {
        btcp_errno = ERR_INIT_CQ_FAIL;
        
        return -1;
    }
    if (!btcp_send_queue_init(&handler->send_buf, DEF_SEND_BUFSZ))
    {
        btcp_errno = ERR_INIT_CQ_FAIL;
        
        return -1;
    }
    btcp_rtt_init(&handler->rtt);
    btcp_timer_init(&handler->timeout);
    btcp_sack_blocklist_init(&handler->sack);
    handler->cong_wnd = 1;
    return 0;
}
int btcp_destroy_tcpconn(struct btcp_tcpconn_handler *handler, bool is_server)
{
    //不要动 peer_ip port字段，这个在服务端的hash table是key，还用用来从hash table里删除的
    if (!is_server)
    {
        if (handler->udp_socket > 0)
        {
            close(handler->udp_socket);
        }
    }
    if (handler->user_socket_pair[0] > 0)
    {
        close(handler->user_socket_pair[0]);
    }
    if (handler->user_socket_pair[1] > 0)
    {
        close(handler->user_socket_pair[1]);
    }
    btcp_recv_queue_destroy(&handler->recv_buf);
    btcp_send_queue_destroy(&handler->send_buf);
    btcp_timer_destroy(&handler->timeout);
    btcp_rtt_destroy(&handler->rtt);
    btcp_sack_blocklist_destroy(&handler->sack);
       
    return 0;

}


static void* btcp_tcpcli_loop(void *arg)
{
    struct btcp_tcpconn_handler *handler = (struct btcp_tcpconn_handler *)arg;
    printf("btcp_tcpcli_loop() start...,  %d, %u\n", sizeof(void*), handler);
    int timeout = 100; // 默认0.1s
    
    static char bigbuffer[1024*64] __attribute__((aligned(8))); //临时用一下，不会跨线程，也不会超出函数的作用域
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);

    while (1)
    {
        int fd_num = 0;
        struct pollfd pfd[2];
        pfd[0].fd = handler->udp_socket;
        pfd[0].events = POLLIN;
        fd_num++;


        if (handler->user_socket_pair[1] >= 0)
        {
            pfd[1].fd = handler->user_socket_pair[1];
            pfd[1].events = POLLIN;
            fd_num++;
        }
        

        int ret = poll(pfd, fd_num, timeout); // 1 秒超时
       
        if (ret > 0)
        {
            if (pfd[0].revents & POLLIN) //底层udp可读，收包并处理
            {
                
                ssize_t received = recvfrom(pfd[0].fd, bigbuffer, sizeof(bigbuffer), 0,
                                            (struct sockaddr *)&client_addr, &addr_len);
                g_info("recv remote data, len=%d\n", received);
                if (received > 0)
                {
                    if (handler->status == SYNC_SENT)
                    {
                        if (btcp_handle_sync_sent(bigbuffer, handler))
                        {
                            printf("btcp_handle_sync_sent() failed, err:%d\n", btcp_errno);
                        }
                    }
                    else if (handler->status != CLOSED)
                    {
                        if (btcp_handle_data_rcvd(bigbuffer, received, handler, &client_addr))
                        {
                            printf("btcp_handle_data_rcvd() failed, err:%d\n", btcp_errno);
                        }
                    }

                }
                else if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    printf("No data available.\n");
                }
            }
            if (fd_num > 1 && pfd[1].revents & POLLIN) //用户层发数据过来了，放置到发送队列里
            {
                if (handler->status == ESTABLISHED || handler->status==CLOSE_WAIT)
                {
                    int space = btcp_send_queue_get_available_space(&handler->send_buf); // 获得发送缓冲区的空闲空间大小
                    //g_info("available space:%d bytes\n", space);
                    if (space > 0 && handler->status == ESTABLISHED)
                    {
                        ssize_t received = read(pfd[1].fd, bigbuffer, space);
                        if (received > 0)
                        {
                            int written = btcp_send_queue_enqueue(&handler->send_buf, bigbuffer, received);
                            g_info("get %d bytes from user, write %d bytes into queue\n", received, written);
                            btcp_try_send(handler);
                        }
                        else if (received == 0)
                        {
                            #if 1
                            //上层应用主动关闭，有两种情况
                            g_info("detect user closed the conn");
                            handler->user_socket_pair[1] = -1;
                            if (handler->status == ESTABLISHED || handler->status == CLOSE_WAIT)
                            {
                               btcp_send_queue_push_fin(&handler->send_buf);
                            }
                            #endif
                        }
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
        
        if (btcp_check_send_timeout(handler))//检查可能的发包超时未ack
        {
            //btcp_try_send(&handler); // 立即（重）发tcp包，因为下面本身也会调用，所以先注释掉
        }
        btcp_try_send(handler); // 尝试发tcp包

        #if 1
        if (btcp_keep_alive(handler, bigbuffer, false) == 1)
        {
            g_info("keepalive close the conn to (%s,%d)", handler->peer_ip, handler->peer_port);
            btcp_destroy_tcpconn(handler, false);
            break;
        }
        #endif
        if (btcp_send_queue_size(&handler->send_buf))
        {
            // 只要还有tcp报文未发送，那么超时时间就极短
            timeout = 0;
        }
        else
        {
            timeout = 100;
        }
    }
    return NULL;
}



int btcp_tcpcli_new_loop_thread(struct btcp_tcpconn_handler *handler)
{
    g_info("in new_loop_thread(), peer ip:%s, mss:%d, peer_port:%d\n", handler->peer_ip, handler->mss, handler->peer_port);
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

static int btcp_tcpsrv_orgnize_poll_fds(struct btcp_tcpsrv_handler * srv, struct pollfd *fds,
                            struct btcp_tcpconn_handler* conns[])
{
    int i = 0;
    fds[i].fd = srv->udp_socket;
    fds[i].events = POLLIN;
    conns[i] = NULL;
    i++;
    

    //  遍历哈希表
    GHashTableIter iter;
    gpointer key, value;

 
    g_hash_table_iter_init(&iter, srv->all_connections);
    
    while (g_hash_table_iter_next(&iter, &key, &value))
    {
        struct btcp_tcpconn_handler *handler = (struct btcp_tcpconn_handler *)value;
        if (handler->status != CLOSED && i < MAX_CONN_ALLOWED)
        {
            fds[i].fd = handler->user_socket_pair[1];
            fds[i].events = POLLIN;

            conns[i] = handler;
            i++; 
            
        }
    }
 
    return i;
}
// srv需要对所有连接做的keep alive
static int btcp_tcpsrv_keep_alive(struct btcp_tcpsrv_handler * srv, char *bigbuffer)
{
    GHashTableIter iter;
    gpointer key, value;
    
     
    g_hash_table_iter_init(&iter, srv->all_connections);
    GList *conns_to_close = NULL;
    while (g_hash_table_iter_next(&iter, &key, &value))
    {
        struct btcp_tcpconn_handler *handler = (struct btcp_tcpconn_handler *)value;
        if (handler->status != CLOSED)
        {
            if (btcp_keep_alive(handler, bigbuffer, true) == 1)
            {
// 这个关闭操作，还不能再这里干，因为是处于迭代器使用中，不能修改hash表
#if 0
                    // btcp_destroy_tcpconn(handler, true); //这个要注释掉，因为hash table在remove的时候会调用释放函数，里面有调用这个函数
                    g_info("keepalive close the conn to (%s,%d)", handler->peer_ip, handler->peer_port);
                    struct btcp_tcpconn_handler *removed = (struct btcp_tcpconn_handler *)g_hash_table_remove(srv->all_connections, handler); // close the connn
#else
                conns_to_close = g_list_insert(conns_to_close, handler, 0);
#endif
            }
        }
    }
    
    GList *iter2;
    for (iter2 = conns_to_close; iter2 != NULL; iter2 = iter2->next)
    {
        struct btcp_tcpconn_handler *handler = (struct btcp_tcpconn_handler *)iter2->data;
        // btcp_destroy_tcpconn(handler, true); //这个要注释掉，因为hash table在remove的时候会调用释放函数，里面有调用这个函数
        g_info("keepalive close the conn to (%s,%d)", handler->peer_ip, handler->peer_port);
        g_hash_table_remove(srv->all_connections, handler); // close the connn
    }
    g_list_free(conns_to_close);
    conns_to_close = NULL;
    return 0;
}
// srv需要对所有连接做的 尝试发包和超时检查
static int btcp_tcpsrv_circular_task(struct btcp_tcpsrv_handler * srv, char *bigbuffer, int *timeout)
{
    GHashTableIter iter;
    gpointer key, value;
    g_hash_table_iter_init(&iter, srv->all_connections);
   
    bool has_data_to_send = false;
    
    while (g_hash_table_iter_next(&iter, &key, &value))
    {
        struct btcp_tcpconn_handler *handler = (struct btcp_tcpconn_handler *)value;
        if (handler->status != CLOSED)
        {
            if (btcp_check_send_timeout(handler)) // 检查可能的发包超时未ack
            {
                // btcp_try_send(&handler); // 立即（重）发tcp包，因为下面本身也会调用，所以先注释掉
            }
            btcp_try_send(handler); // 尝试发tcp包
            
            if (btcp_send_queue_size(&handler->send_buf))
            {
                // 只要还有tcp报文未发送，那么超时时间就极短
                has_data_to_send = true;
            }
        
        }
    }
    
    if (has_data_to_send)
    {
        *timeout = 0;
    }
    else
    {
        *timeout = 10;
    }
    return 0;
}

static void* btcp_tcpsrv_loop(void * arg)
{
    struct btcp_tcpsrv_handler * srv = (struct btcp_tcpsrv_handler*)arg;
    static char bigbuffer[100*1024]  __attribute__((aligned(8)));//用于收发包临时数据存储，不会跨线程、不会跨连接
    int timeout = 100;
    
    while (1)
    {
        struct pollfd fds[MAX_CONN_ALLOWED+1]; // 0号fd是底层udp，其他都是socketpair与用户层通信
        struct btcp_tcpconn_handler* conns[MAX_CONN_ALLOWED+1]; // 与上面按下标一一对应
        int fd_num = btcp_tcpsrv_orgnize_poll_fds(srv, fds, conns);
        int ret = poll(fds, fd_num, timeout); 
        
        if (ret > 0)
        {
            if (fds[0].revents & POLLIN) //底层udp可读，收包并处理
            {
                struct sockaddr_in client_addr;
                socklen_t addr_len = sizeof(struct sockaddr);
                int pkg_len = recvfrom(fds[0].fd, bigbuffer, sizeof(bigbuffer), 0,
                                        (struct sockaddr *)&client_addr, &addr_len);
                if (pkg_len > 0)
                {
                    char ip_str[INET_ADDRSTRLEN]; // 用于存储IP地址字符串
                    inet_ntop(AF_INET, &client_addr.sin_addr, ip_str, INET_ADDRSTRLEN);

                    unsigned short dest_p, source_p;
                    btcp_get_port(bigbuffer, &dest_p, &source_p);

                    // 对于复杂结构体或者字符串类型的key，
                    // g_hash_table_lookup() 和 g_hash_table_remove() 的 key 参数 可以是栈上分配的内存的指针，
                    // 因为这两个函数 不会记录或引用 key 参数的内存。它们只是使用 key 参数的内容来计算哈希值并进行
                    // 查找或删除操作。但g_hash_table_insert()函数的key就不能分配在栈上。
                    struct btcp_tcpconn_handler key; // 用于查找hash table
                    strncpy(key.peer_ip, ip_str, INET_ADDRSTRLEN);
                    key.peer_port = source_p;

                    struct btcp_tcpconn_handler *conn = g_hash_table_lookup(srv->all_connections, &key);
                    if (conn == NULL) // 没有就创建并插入
                    {
                        if (g_hash_table_size(srv->all_connections) >= MAX_CONN_ALLOWED)
                        {
                            g_warning("max conn allowed reached!");
                            continue;
                        }

                        conn = btcp_handle_sync_rcvd1(bigbuffer, srv, &client_addr);
                        if (conn == NULL)
                        {
                            fprintf(stderr, "btcp_handle_sync_rcvd1() failed! %d\n", btcp_errno);
                            continue;
                        }
                        pthread_mutex_lock(&srv->all_connections_mutex);
                        if (!g_hash_table_insert(srv->all_connections, conn, conn)) // 键值都是conn，注意。
                        {
                            g_warning("!!!g_hash_table_insert() failed");
                            pthread_mutex_unlock(&srv->all_connections_mutex);
                            continue;
                        }
                        pthread_mutex_unlock(&srv->all_connections_mutex);
                    }
                    else if (conn->status == SYNC_RCVD)
                    {
                        if (btcp_handle_sync_rcvd2(bigbuffer, conn, &client_addr))
                        {
                            fprintf(stderr, "btcp_handle_sync_rcvd2() failed! %d\n", btcp_errno);
                            pthread_mutex_lock(&srv->all_connections_mutex);
                            g_hash_table_remove(srv->all_connections, &key); // close the connn
                            pthread_mutex_unlock(&srv->all_connections_mutex);
                        }
                    }
                    else if (conn->status != CLOSED)
                    {
                        if (btcp_handle_data_rcvd(bigbuffer, pkg_len, conn, &client_addr))
                        {
                            fprintf(stderr, "btcp_handle_data_rcvd() failed! %d\n", btcp_errno);
                            // 这里还是不删除比较好，避免恶意者攻击
                            // struct btcp_tcpconn_handler * removed =  (struct btcp_tcpconn_handler *)g_hash_table_remove(srv->all_connections, &key); // close the connn // close the connn
                        }
                    }
                }
                
            }
            for (int i = 1; i < fd_num; ++i) //循环监视所有的用户层fd，fd很大时要改为epoll性能才好些
            {
                
                if (fds[i].revents & POLLIN)
                {
                    
                    struct btcp_tcpconn_handler * handler = conns[i];
                    if (handler->status == CLOSED)
                    {
                        continue;
                    }
                    
                    int space = btcp_send_queue_get_available_space(&handler->send_buf); // 获得发送缓冲区的空闲空间大小
                    if (space > 0)
                    {
                        
                        ssize_t received = read(fds[i].fd, bigbuffer, space);
                        if (received > 0)
                        {
                            int written = btcp_send_queue_enqueue(&handler->send_buf, bigbuffer, received);
                            g_info("get %d bytes from user, write %d bytes into queue\n", received, written);
                            btcp_try_send(handler);
                        }
                        else if (received == 0) //用户层关闭了socketpair，需要对tcpconn做FIN
                        {
                            //上层应用主动关闭，有两种情况
                            //g_info("server side, user closed the conn");
                            if (handler->status == ESTABLISHED || handler->status == CLOSE_WAIT)
                            {
                               btcp_send_queue_push_fin(&handler->send_buf);
                            }
                        }
                    }
                }
            }
        }
        //到这里，poll就处理完了
        
        //  遍历哈希表, 做一些定时要做的事情
        // 1) keep alive
        btcp_tcpsrv_keep_alive(srv, bigbuffer);
        // 2) 尝试发数据，检查发包超时没有
        btcp_tcpsrv_circular_task(srv, bigbuffer, &timeout);
    }
}

int btcp_tcpsrv_new_loop_thread(struct btcp_tcpsrv_handler * srv)
{
    pthread_t thread_id;
    

    // 创建线程
    if (pthread_create(&thread_id, NULL, btcp_tcpsrv_loop, (void *)srv) != 0) {
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

GList *  btcp_tcpsrv_get_all_conn_fds(struct btcp_tcpsrv_handler * srv, int * status)
{
    GList * result = NULL;
    // 遍历哈希表
    GHashTableIter iter;
    gpointer key, value;

    pthread_mutex_lock(&srv->all_connections_mutex);
    g_hash_table_iter_init(&iter, srv->all_connections);
    
    while (g_hash_table_iter_next(&iter, &key, &value))
    {
        struct btcp_tcpconn_handler * handler = (struct btcp_tcpconn_handler *)value;
        if (status != NULL && handler->status != *status) //如果指定了需要返回某种状态的连接
        {
            continue;
        }
        

        result = g_list_insert(result, GINT_TO_POINTER(handler->user_socket_pair[0]), 0);
    }
    pthread_mutex_unlock(&srv->all_connections_mutex);
    return result;
}

/*
static void free_one_conn(gpointer data) {
    struct btcp_tcpconn_handler * conn = (struct btcp_tcpconn_handler *)data;
    if (conn != NULL) {
        free(conn);
    }
}
*/

void btcp_free_conns_in_glist(GList * conns)
{
    if (conns == NULL) {return;}
    /*
    for (const GList *iter = conns; iter != NULL; iter = iter->next) {
        struct btcp_tcpconn_handler *conn = (struct btcp_tcpconn_handler *)iter->data;
        free(conn);
    }
    */
    g_list_free(conns);
}
