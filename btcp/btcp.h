#ifndef BTCP_H_INCLUDED

#define BTCP_H_INCLUDED

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "btcp_send_queue.h"
#include "btcp_timeout.h"
#include "circular_queue.h"

extern int btcp_errno;
int btcp_get_route_mtu(const char *dest_ip);
int btcp_alloc_local_port();
int btcp_free_local_port(unsigned short port);
unsigned int btcp_get_random();


#define DEF_RECV_BUFSZ (4096)
#define DEF_SEND_BUFSZ (4096)


#define ERR_GET_MTU_FAIL  (-100)
#define ERR_INIT_UDP_FAIL  (-101)
#define ERR_INIT_CQ_FAIL  (-102)
#define ERR_GET_LPORT_FAIL  (-103)
#define ERR_UDP_COMMM_FAIL  (-104)
#define ERR_INVALID_PKG  (-105)
#define ERR_SEQ_MISMATCH (-106)
#define ERR_PORT_MISMATCH (-107)
#define ERR_INVALID_ARG (-108)
#define ERR_UDP_PORT_USED (-109)
#define ERR_MEM_ERROR (-110)

enum btcp_tcpconn_status
{
    CLOSED,
    SYNC_SENT,
    SYNC_RCVD,
    ESTABLISHED
};

struct btcp_tcpsrv_handler
{
    int udp_socket;
    char my_ip[INET_ADDRSTRLEN];

    int local_port;
};

struct btcp_tcpconn_handler
{
    int udp_socket;
    char peer_ip[INET_ADDRSTRLEN];
    char local_ip[INET_ADDRSTRLEN];

    int my_recv_wnd_sz;
    int peer_recv_wnd_sz;
    int cong_wnd;
    int cong_wnd_threshold;

    int mss;
    uint32_t local_seq; //发送窗口（允许未被确认的字节段）的第一个字节编号
    uint32_t peer_seq; //到目前为止已经可以确认收到的对端的sequence， 我端发出报文的ack seq等于peer_seq+1
    int local_port;
    int peer_port;
    enum btcp_tcpconn_status status;
    
    btcp_circular_queue recv_buf;

    struct btcp_send_queue send_buf;
    struct btcp_timeout timeout; //发送的报文的超时控制

    int user_socket_pair[2]; // unix domain socket，用来和上层应用进行收发数据, 上层应用读写user_socket_pair[0], btcp读写user_socket_pair[1]，他们是相连的一对
};
struct btcp_tcphdr
{
    uint16_t source;      // 源端口号
    uint16_t dest;        // 目的端口号
    uint32_t seq;         // 序列号
    uint32_t ack_seq;     // 确认号
/*
(TCP Offset, Reserved, Flags)， 对应主机上一个uint16_t的各个位如下：

| 位索引 (Bit Index) | 15 - 12 (4 位) | 11 - 9 (3 位)  | 8 - 0 (9 位)      |
|--------------------|----------------|----------------|-------------------|
| 字段               | Offset         | Reserved       | Flags            |

字段说明：
- Offset  : 表示 TCP 首部长度，单位是 4 字节。
- Reserved: 预留字段，通常置 0，用于未来扩展。
- Flags   : TCP 控制位，包括 ACK, SYN, FIN 等标志。

*/
    uint16_t doff_res_flags; // 包含数据偏移、保留位和标志位
    uint16_t window;         // 窗口大小
    uint16_t check;          // 校验和
    uint16_t urg_ptr;        // 紧急指针  
};
union btcp_tcphdr_with_option
{
    struct btcp_tcphdr base_hdr; //20Byte
    // 如果需要处理选项字段，可以额外定义一个选项数组
    uint8_t options[sizeof(struct btcp_tcphdr)+40];  // TCP 选项字段（可选，最长 40 字节）
};
enum btcp_tcphdr_flag
{
    FLAG_FIN = 0,
    FLAG_SYN = 1,
    FLAG_RST = 2,
    FLAG_PSH,
    FLAG_ACK,
    FLAG_URG,
    FLAG_ECE,
    FLAG_CWR,
    FLAG_NS
} ;
int btcp_check_tcphdr_flag(enum btcp_tcphdr_flag flag, uint16_t doff_res_flags);
int btcp_clear_tcphdr_flag(enum btcp_tcphdr_flag flag, uint16_t * doff_res_flags);
int btcp_set_tcphdr_flag(enum btcp_tcphdr_flag flag, uint16_t * doff_res_flags);

int btcp_set_tcphdr_offset(int offset, uint16_t * doff_res_flags);
int btcp_get_tcphdr_offset(const uint16_t * doff_res_flags);

int btcp_set_socket_nonblock(int sockfd);
int btcp_is_readable(int sockfd, int to, char * bigbuffer, int buflen, struct sockaddr_in *client_addr);
int btcp_get_port(const char*bigbuffer, unsigned short * dest, unsigned short *source);
int btcp_print_tcphdr(const char*bigbuffer, const char * msg);

#endif