#ifndef BTCP_H_INCLUDED

#define BTCP_H_INCLUDED

/*
 * 这是一个重要且信息很多的头文件，引擎和开发者都需要引用到。
 * 开发者是通过 #include "btcp_api.h"文件引用到该文件的
 */

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
#include "btcp_recv_queue.h"
#include <glib.h>
#include "tool.h"
#include "btcp_selective_ack_blocklist.h"

#include "btcp_rtt.h"

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

// tcp连接所处的各种状态
enum btcp_tcpconn_status
{
    CLOSED = 0,
    SYNC_SENT,
    SYNC_RCVD,
    ESTABLISHED,

    FIN_WAIT1,
    FIN_WAIT2,
    TIME_WAIT,

    CLOSE_WAIT,
    LAST_ACK

};

//一个tcpsrv的句柄
struct btcp_tcpsrv_handler
{
    int udp_socket;
    char my_ip[INET_ADDRSTRLEN];

    int local_port;

    // 用户线程和引擎线程可能同时访问下面的hash表，用这个来互斥：
    // 引擎修改hashtable的时候要上锁
    // 用户线程迭代器获得所有hashtable中的value的时候要上锁
    pthread_mutex_t all_connections_mutex; 
    GHashTable * all_connections; //保存了所有tcp连接的hash表
};

//一条tcp连接的句柄
struct btcp_tcpconn_handler
{
    int udp_socket;
    char peer_ip[INET_ADDRSTRLEN];
    char local_ip[INET_ADDRSTRLEN];

    
    int peer_recv_wnd_sz; //对端接收窗口的大小，单位为byte
    int cong_wnd;         // 本端拥塞窗口的大小，单位为mss
    int cong_wnd_threshold; // 拥塞算法中用到的一个窗口阈值，小于它时指数增长，大于它后线性增长

    //cong_wnd上一次加一的时间，用于超过threshold
    // 后每个rtt轮次加一，简单起见，我这里实现为每秒钟加一 
    time_t cong_wnd_prev_inc_time; 

    int repeat_ack; //当发生丢包的时候，反复收到对端对当前窗口起始sequence的ack，超过3次就重发并收缩窗口
    time_t alive_time_stamp; //保活时间戳 记录本连接最后一次活跃时刻
    time_t keepalive_request_time; //上一次发送keepalive请求的时间，记录下来避免频繁发送

    int mss;
    struct btcp_rtt_handler rtt; // 用来粗略统计本连接rtt的一个模块
    struct btcp_sack_blocklist sack;
    uint32_t local_seq; //发送窗口（允许未被确认的字节段）的第一个字节编号
    uint32_t peer_seq; //期望收到对端发的顺序包的起始sequence，
    int local_port;
    int peer_port;
    enum btcp_tcpconn_status status; //连接当前的状态，ESTABLISHED or CLOSED etc.
    
    struct btcp_recv_queue recv_buf; //接收缓冲区
    struct btcp_send_queue send_buf; //发送缓冲区
    struct btcp_timeout timeout; //发送的报文的超时控制，可以理解为一个链表

    // unix domain socket，用来和上层应用进行收发数据, 上层应用读写user_socket_pair[0], 
    // btcp读写user_socket_pair[1]，他们是相连的一对
    int user_socket_pair[2]; 
};

// tcp首部
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

// 因为tcp首部可能包含不定长的options，编程方便起见，定义这样一个union
union btcp_tcphdr_with_option
{
    struct btcp_tcphdr base_hdr; //20Byte
    // 如果需要处理选项字段，通过下面的变量进行访问
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


//检查tcp首部flag字段是否设置了某个上述的flag
int btcp_check_tcphdr_flag(enum btcp_tcphdr_flag flag, uint16_t doff_res_flags);

//将tcp首部清理掉某个上述的flag
int btcp_clear_tcphdr_flag(enum btcp_tcphdr_flag flag, uint16_t * doff_res_flags);

//将tcp首部设置上某个上述的flag
int btcp_set_tcphdr_flag(enum btcp_tcphdr_flag flag, uint16_t * doff_res_flags);

// 为tcp首部设置用户数据偏移，也就是用户数据开始的位置
int btcp_set_tcphdr_offset(int offset, uint16_t * doff_res_flags);

// 从tcp首部获取用户数据偏移，也就是用户数据开始的位置
int btcp_get_tcphdr_offset(const uint16_t * doff_res_flags);

// 从报文中获取目的端口和发送端口，报文包含tcp首部，存储在bigbuffer中
int btcp_get_port(const char*bigbuffer, unsigned short * dest, unsigned short *source);

// 打印tcp首部的各个字段，调试用
int btcp_print_tcphdr(const char*bigbuffer, const char * msg);






#endif