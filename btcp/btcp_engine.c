#include "btcp.h"
#include "btcp_engine.h"
#include <poll.h>
#include <stdint.h>


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
/*
在 GLib 的 GHashTable 中，g_hash_table_new_full() 指定的释放函数（key_destroy_func 和 value_destroy_func）不会在 g_hash_table_remove() 执行时被自动调用。这些释放函数只会在以下情况下被调用：

哈希表被销毁时（调用 g_hash_table_destroy()）：

哈希表中所有剩余的键值对会被释放，key_destroy_func 和 value_destroy_func 会被调用。

键被替换时（调用 g_hash_table_insert() 或 g_hash_table_replace() 插入一个已存在的键）：

如果插入的键已经存在，旧的键值对会被替换，key_destroy_func 和 value_destroy_func 会被调用以释放旧的键和值。

g_hash_table_remove() 的行为
g_hash_table_remove() 只会从哈希表中移除键值对，并返回被移除的值。

它不会调用 key_destroy_func 或 value_destroy_func。

如果键和值是动态分配的内存，你需要手动释放它们。
*/
// 释放键的函数
static void btcp_tcpconn_key_destroy(gpointer key) {

    struct btcp_tcpconn_handler* k = (struct btcp_tcpconn_handler*)key;
    
    g_info("Destroying key: ip=%s, port=%d", k->peer_ip, k->peer_port);
    btcp_destroy_tcpconn(k, true);
    free(key);  // 释放动态分配的结构体内存
}

// 释放值的函数
static void btcp_tcpconn_value_destroy(gpointer value) {
   struct btcp_tcpconn_handler* v = (struct btcp_tcpconn_handler*)value;
   g_info("Destroying value: ip=%s, port=%d", v->peer_ip, v->peer_port);
    // 注意：如果键和值是同一个指针，这里不需要释放内存， 也不需要做 btcp_destroy_tcpconn
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
        memset(handler, 0, sizeof(struct btcp_tcpconn_handler));
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
        //测试用：
        handler->mss = 10;
        if (!btcp_recv_queue_init(&handler->recv_buf, DEF_RECV_BUFSZ))
        {
            btcp_errno = ERR_INIT_CQ_FAIL;
            free(handler);
            return NULL;
        }
        if (!btcp_send_queue_init(&handler->send_buf, DEF_SEND_BUFSZ))
        {
            btcp_errno = ERR_INIT_CQ_FAIL;
            free(handler);
            return NULL;
        }
        handler->cong_wnd = 1;
        handler->local_recv_wnd_sz = DEF_RECV_BUFSZ;

       
        handler->local_port = srv->local_port;
        
        handler->local_seq = btcp_get_random();
        
        handler->status = SYNC_RCVD;
        handler->udp_socket = srv->udp_socket;
    }

    handler->peer_seq = btcp_sequence_step_forward(ntohl(hdr->seq), 1);
    btcp_recv_queue_set_expected_seq(&handler->recv_buf, handler->peer_seq);
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
    handler->local_seq++;

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
    
    btcp_set_socket_nonblock(handler->user_socket_pair[0]);
    btcp_set_socket_nonblock(handler->user_socket_pair[1]);

    //收到了对端的合法报文，认为该连接处于活跃状态，更新保活时间戳
    handler->alive_time_stamp = time(NULL);
    
    handler->status = ESTABLISHED;
    printf("established!\n");
    return 0;
}

int btcp_handle_ack(union btcp_tcphdr_with_option *tcphdr, struct btcp_tcpconn_handler *handler)
{
    struct btcp_tcphdr * hdr = &tcphdr->base_hdr;
    uint32_t ack_seq32 = ntohl(hdr->ack_seq);
    uint64_t ack_seq64 = ack_seq32;

    
    if (ack_seq32 < handler->local_seq) //发生了回绕
    {
        ack_seq64 = btcp_sequence_round_out(ack_seq32);
    }
    
    if (ack_seq64 > (handler->local_seq + 65535)) // 大太多了，就算是累计确认也不能差这么多
    {
        g_error("ack sequence is too big! %u, %u", ack_seq32, handler->local_seq);
        return -1;
    }
    if (handler->local_seq == ack_seq32) // 收到对当前sequence的重复确认
    {
        handler->repeat_ack++;
        if (handler->repeat_ack >= 3) //连续收到3次或者以上，触发窗口缩小和重发
        {
            handler->repeat_ack = 0;
            // 修改发送窗口大小
            handler->cong_wnd_threshold = handler->cong_wnd / 2;
            
            if (handler->cong_wnd_threshold < 4)
            {
                handler->cong_wnd_threshold = 4;
            }
            handler->cong_wnd = handler->cong_wnd_threshold;
            
            btcp_timer_remove_by_from(&handler->timeout, handler->local_seq);//删除计时器里起始seq等于local_seq的记录
            //btcp_try_send(handler); //删掉计时器里的记录，其实后面就会比较及时的重发
        }
    }
    else
    {
        handler->repeat_ack = 0;

        struct btcp_range range;
        range.from = handler->local_seq;
        range.to = ack_seq64;

        handler->local_seq = ack_seq32;
        btcp_send_queue_set_start_seq(&handler->send_buf, ack_seq64);
        g_info("local sequence step forward to %u", handler->local_seq);
        // 删除可能的定时器
        btcp_timer_remove_range(&handler->timeout, &range);

        handler->cong_wnd++;
        if (handler->cong_wnd > handler->cong_wnd_threshold)
        {
            handler->cong_wnd = handler->cong_wnd_threshold;
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
            hdr->window = htons(DEF_RECV_BUFSZ); // todo:修改为当前接收缓冲区实际的可用空间

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
        g_error("unexpected data size! %d, %s %d", size, __FILE__, __LINE__);
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
        int written = write(handler->user_socket_pair[1], buf, len);
        if (written!= len)
        {
            g_error("write user socket pair failed!%d, (%s, %d)", written, __FILE__, __LINE__);
        }
        g_info("thow %d bytes data to users", written);
    }
    return 0;
}
int btcp_handle_data_rcvd(char * bigbuffer, int pkg_len, struct btcp_tcpconn_handler * handler, 
            const struct sockaddr_in * client_addr)
{
    
    union btcp_tcphdr_with_option *tcphdr = (union btcp_tcphdr_with_option *)bigbuffer;
    struct btcp_tcphdr * hdr = &tcphdr->base_hdr;

    btcp_print_tcphdr((const char *)hdr, "recv user data:");

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
      
    if (data_len > 0)
    {
        uint64_t from_seq = ntohl(hdr->seq);
        uint64_t to_seq = from_seq + data_len - 1;
        // save data to recv queue
        if (btcp_recv_queue_save_data(&handler->recv_buf, from_seq, to_seq,
                                      bigbuffer + offset))
        {
            btcp_errno = ERR_SEQ_MISMATCH;
            return -1;
        }
        g_info("data_len:%d, peer_seq:%u", data_len, handler->peer_seq);
        
        if ((handler->peer_seq ) == ntohl(hdr->seq)) // 收到了想要的下一个（顺序）报文，需要移动接收窗口
        {
            //移动的大小不一定就等于data_len，因为可能之前已经收到过 后发先至 的数据段，与这个报文连成一片。
            int steps = btcp_recv_queue_try_move_wnd(&handler->recv_buf);
            if (steps < 0)
            {
                g_error("btcp_recv_queue_try_move_wnd() failed! %d", steps);
                btcp_errno = ERR_SEQ_MISMATCH;
                return -1;
            }
            g_info("recv wnd move %d bytes", steps);

            // sequence step forward
            handler->peer_seq = btcp_sequence_step_forward(handler->peer_seq, steps);
            //本来这里也需要同步的修改recv queue里的expected_seq，但try_move_wnd函数里面已经修改了

            g_info("peer_seq changes to:%u, expected_seq:%u, tail:%d",  handler->peer_seq,
                    handler->recv_buf.expected_seq,
                    handler->recv_buf.tail);
            //向应用层抛数据
            btcp_throw_data_to_user(handler);
            
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
        hdr->window = htons(DEF_RECV_BUFSZ);// todo:修改为当前接收缓冲区实际的可用空间

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
    }

    //收到了对端的合法报文，认为该连接处于活跃状态，更新保活时间戳
    handler->alive_time_stamp = time(NULL);
    g_info("modify alive timestamp to current");


    if (btcp_check_tcphdr_flag(FLAG_FIN, (hdr->doff_res_flags)) ) // 如果带有fin标记
    {
        // todo:process fin request
        // btcp_handle_fin(tcphdr, handler);
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
    //测试用：
    handler->mss = 10;
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
    handler->cong_wnd = 1;
    handler->cong_wnd_threshold = 8;
    handler->local_recv_wnd_sz = DEF_RECV_BUFSZ;
    
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
        handler->local_seq = btcp_get_random()%UINT16_MAX;
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
    btcp_recv_queue_set_expected_seq(&handler->recv_buf, handler->peer_seq);
    uint32_t ack_seq = ntohl(hdr->ack_seq);
    if (ack_seq != (handler->local_seq + 1) )
    {
        btcp_errno = ERR_SEQ_MISMATCH;
        return -1;
    }
    handler->peer_recv_wnd_sz = ntohs(hdr->window);


    handler->local_seq++;
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
    hdr->window = htons(DEF_RECV_BUFSZ);
    
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
    /*
    g_info("send_wndsz:%d, mss:%d, cong_wnd:%d, peer wndsz:%d", 
            send_wndsz,
            handler->mss,
            handler->cong_wnd,
            handler->peer_recv_wnd_sz);
            */
    int datasz_in_queue = btcp_send_queue_size(&handler->send_buf);
    if (datasz_in_queue < 1)
    {
        return 0;
    }
    //g_info("尝试发送数据，窗口大小为%d bytes, 发送缓冲里的数据有 %d bytes\n", send_wndsz, datasz_in_queue);
    //发送窗口的范围，与已经发送的待ack报文覆盖的范围比较，找出需要发送的数据段 ，
    //这里参与运算的seq/from/to使用uint64_t类型，且保证to >= from，即to可能大于UINT32_MAX
    struct btcp_range* range_to_send = malloc(sizeof(struct btcp_range));
    range_to_send->from = handler->send_buf.start_seq;
    if (send_wndsz <= datasz_in_queue)
    {
        range_to_send->to = (uint64_t)(handler->send_buf.start_seq) + send_wndsz - 1; //闭区间，所以要减一
    }
    else
    {
        range_to_send->to = (uint64_t)(handler->send_buf.start_seq) + datasz_in_queue - 1; //闭区间，所以要减一
    }
    //g_info("data range to send:[%llu, %llu]\n", range_to_send->from, range_to_send->to);
    
    GList *range_list_to_send = NULL;
    range_list_to_send = g_list_append(NULL, range_to_send);
    
    GList *range_list_sent = NULL;
    if (btcp_timer_get_all_event(&handler->timeout, &range_list_sent) != 0)
    {
        btcp_errno = ERR_MEM_ERROR;
        goto btcp_try_send_out;
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
    GList * range_list_result = NULL, *combined_list = NULL;
    if (btcp_range_subtract(range_list_to_send, range_list_sent, &range_list_result))
    {
        btcp_errno = ERR_MEM_ERROR;
        goto btcp_try_send_out;
    }
    btcp_range_list_combine(range_list_result, &combined_list);
    {
        #ifdef _DETAIL_LOG_
        g_info("data to send:");
        for (const GList *iter = combined_list; iter != NULL; iter = iter->next)
        {
            struct btcp_range *a_range = (struct btcp_range *)iter->data;
            g_info("[%llu, %llu]", a_range->from, a_range->to);
        }
        #endif
    }
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
            if (btcp_send_queue_fetch_data(&handler->send_buf, b_range.from, b_range.from + datalen - 1, bigbuffer+sizeof(struct btcp_tcphdr)))
            {
                g_error("!!!btcp_send_queue_fetch_data() failed\n");
                break;
            }
            g_info("send a tcp package[%llu, %llu]\n", b_range.from, b_range.from + datalen - 1);

            
            struct btcp_tcphdr * hdr = (struct btcp_tcphdr *)bigbuffer;
            memset(hdr, 0, sizeof(struct btcp_tcphdr));
            hdr->dest = htons(handler->peer_port);
            hdr->source = htons(handler->local_port);
            hdr->window = htons(DEF_RECV_BUFSZ);
            hdr->seq = htonl( btcp_sequence_round_in(b_range.from));
            int offset = sizeof(struct btcp_tcphdr);
            btcp_set_tcphdr_offset(offset, &hdr->doff_res_flags);
            offset += datalen;

            //模拟30%的丢包率 . todo:要改回去
            unsigned int r = btcp_get_random() % 3;
            int sent_len;
            if (r != 0)
            {
                sent_len = sendto(handler->udp_socket, hdr, offset, 0, (const struct sockaddr *)&server_addr, sizeof(server_addr));
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
            }
            else
            {
                g_info("package lost![%llu, %llu]", b_range.from, b_range.to);
                sent_len = offset;
            }
            //g_info("sent successfully, len:%d\n", sent_len);
            // 记录超时事件, timer里记录的range的sequence都是32bit范围内的值，方便与ack报文的sequence对应
            struct btcp_range c_range;
            c_range.from = btcp_sequence_round_in(b_range.from);
            c_range.to =  btcp_sequence_round_in(b_range.from + datalen - 1);   

            if (btcp_timer_add_event(&handler->timeout, 5, &c_range, sizeof(c_range), btcp_range_cmp))
            {
                g_error("登记定时器失败， btcp_timer_add_event() failed!\n");
                break;
            }

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
    while ( btcp_timer_check(&handler->timeout, &e, &len) == 1)
    {
        g_info("ack timeout, [%llu, %llu]\n", e.from, e.to);
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

    
    return 0;

}


static void* btcp_tcpcli_loop(void *arg)
{
    struct btcp_tcpconn_handler *handler = (struct btcp_tcpconn_handler *)arg;
    printf("btcp_tcpcli_loop() start...,  %d, %u\n", sizeof(void*), handler);
    int timeout = 100; // 默认0.1s
    
    static char bigbuffer[1024*64] __attribute__((aligned(8))); 
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);

    while (1)
    {
        struct pollfd pfd[2];
        pfd[0].fd = handler->udp_socket;
        pfd[0].events = POLLIN;

        pfd[1].fd = handler->user_socket_pair[1];
        pfd[1].events = POLLIN;

        int ret = poll(pfd, 2, timeout); // 1 秒超时
       
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
                    else if (handler->status == ESTABLISHED)
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
            if (pfd[1].revents & POLLIN) //用户层发数据过来了，放置到发送队列里
            {
                int space = btcp_send_queue_get_available_space(&handler->send_buf); // 获得发送缓冲区的空闲空间大小
                g_info("available space:%d bytes\n", space);
                if (space > 0 && handler->status == ESTABLISHED)
                {
                    ssize_t received = read(pfd[1].fd, bigbuffer, space);
                    if (received > 0)
                    {
                        int written = btcp_send_queue_enqueue(&handler->send_buf, bigbuffer, received);
                        g_info("get %d bytes from user, write %d bytes into queue\n", received, written);
                        btcp_try_send(handler);
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
        if (btcp_keep_alive(handler, bigbuffer, false) == 1)
        {
            g_info("keepalive close the conn to (%s,%d)", handler->peer_ip, handler->peer_port);
            btcp_destroy_tcpconn(handler, false);
            break;
        }
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

static void* btcp_tcpsrv_loop(void * arg)
{
    struct btcp_tcpsrv_handler * srv = (struct btcp_tcpsrv_handler*)arg;
    static char bigbuffer[100*1024]  __attribute__((aligned(8)));
    struct sockaddr_in client_addr;
    while (1)
    {
        int pkg_len = btcp_is_readable(srv->udp_socket, 100, bigbuffer, sizeof(bigbuffer), &client_addr);
        if (pkg_len > 0)
        {
            char ip_str[INET_ADDRSTRLEN]; // 用于存储IP地址字符串
            inet_ntop(AF_INET, &client_addr.sin_addr, ip_str, INET_ADDRSTRLEN);
            
            
            unsigned short dest_p, source_p;
            btcp_get_port(bigbuffer, &dest_p, &source_p);

            //对于复杂结构体或者字符串类型的key，
            //g_hash_table_lookup() 和 g_hash_table_remove() 的 key 参数 可以是栈上分配的内存的指针，
            //因为这两个函数 不会记录或引用 key 参数的内存。它们只是使用 key 参数的内容来计算哈希值并进行
            //查找或删除操作。但g_hash_table_insert()函数的key就不能分配在栈上。
            struct btcp_tcpconn_handler key;//用于查找hash table
            strncpy(key.peer_ip, ip_str, INET_ADDRSTRLEN);
            key.peer_port = source_p;

            struct btcp_tcpconn_handler *conn = g_hash_table_lookup(srv->all_connections, &key);
            if (conn == NULL) //没有就创建并插入
            {
                if (g_hash_table_size(srv->all_connections) > MAX_CONN_ALLOWED)
                {
                    g_warning("max conn allowed reached!");
                    continue;
                }
                
                conn = btcp_handle_sync_rcvd1(bigbuffer,  srv, &client_addr);
                if (conn == NULL)
                {
                    fprintf(stderr, "btcp_handle_sync_rcvd1() failed! %d\n", btcp_errno);
                    continue;
                }
                if (!g_hash_table_insert(srv->all_connections, conn, conn)) // 键值都是conn，注意。
                {
                    g_error("!!!g_hash_table_insert() failed");
                    continue;
                }
            }
            else if (conn->status == SYNC_RCVD)
            {
                if (btcp_handle_sync_rcvd2(bigbuffer,  conn, &client_addr))
                {
                    fprintf(stderr, "btcp_handle_sync_rcvd2() failed! %d\n", btcp_errno);

                    struct btcp_tcpconn_handler * removed =  (struct btcp_tcpconn_handler *)g_hash_table_remove(srv->all_connections, &key); // close the connn
                    
                }
            }
            else if (conn->status == ESTABLISHED)
            {
                if (btcp_handle_data_rcvd(bigbuffer, pkg_len, conn, &client_addr))
                {
                    fprintf(stderr, "btcp_handle_data_rcvd() failed! %d\n", btcp_errno);
                    struct btcp_tcpconn_handler * removed =  (struct btcp_tcpconn_handler *)g_hash_table_remove(srv->all_connections, &key); // close the connn // close the connn
                    
                }
            }


        }
        
        //  遍历哈希表, 做一些定时要做的事情
        GHashTableIter iter;
        gpointer key, value;
        g_hash_table_iter_init(&iter, srv->all_connections);
        GList * conns_to_close = NULL;
        while (g_hash_table_iter_next(&iter, &key, &value))
        {
            struct btcp_tcpconn_handler *handler = (struct btcp_tcpconn_handler *)value;
            if (handler->status != CLOSED)
            {
                if (btcp_keep_alive(handler, bigbuffer, true) == 1)
                {
                    //这个关闭操作，还不能再这里干，因为是处于迭代器使用中，不能修改hash表
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
        GList * iter2;
        for (iter2 = conns_to_close; iter2 != NULL; iter2 = iter2->next)
        {
            struct btcp_tcpconn_handler *handler = (struct btcp_tcpconn_handler *)iter2->data;
            // btcp_destroy_tcpconn(handler, true); //这个要注释掉，因为hash table在remove的时候会调用释放函数，里面有调用这个函数
            g_info("keepalive close the conn to (%s,%d)", handler->peer_ip, handler->peer_port);
            g_hash_table_remove(srv->all_connections, handler); // close the connn
        }
        g_list_free(conns_to_close);
        conns_to_close = NULL;
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

GList *  btcp_tcpsrv_get_all_connections(struct btcp_tcpsrv_handler * srv, int * status)
{
    GList * result = NULL;
    // 遍历哈希表
    GHashTableIter iter;
    gpointer key, value;
    g_hash_table_iter_init(&iter, srv->all_connections);
    while (g_hash_table_iter_next(&iter, &key, &value))
    {

        if (status != NULL && ((struct btcp_tcpconn_handler *)value)->status != *status)
        {
            continue;
        }

        struct btcp_tcpconn_handler *conn = (struct btcp_tcpconn_handler *)malloc(sizeof(struct btcp_tcpconn_handler));
        if (conn == NULL)
        {
            btcp_errno = ERR_MEM_ERROR;
            break;
        }
        memcpy(conn, value, sizeof(struct btcp_tcpconn_handler));

        result = g_list_insert(result, conn, 0);
    }
    return result;
}

static void free_one_conn(gpointer data) {
    struct btcp_tcpconn_handler * conn = (struct btcp_tcpconn_handler *)data;
    if (conn != NULL) {
        free(conn);
    }
}

void btcp_free_conns_in_glist(GList * conns)
{
    if (conns == NULL) {return;}
    for (const GList *iter = conns; iter != NULL; iter = iter->next) {
        struct btcp_tcpconn_handler *conn = (struct btcp_tcpconn_handler *)iter->data;
        free(conn);
    }
    g_list_free(conns);
}
