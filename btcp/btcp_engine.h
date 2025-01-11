#ifndef BTCP_ENGINE_H_INCLUDED
#define BTCP_ENGINE_H_INCLUDED

#include "btcp.h"
#include "btcp_timeout.h"
#include "tool.h"
#include "btcp_send_queue.h"
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <glib.h>
#include <string.h>
#include <errno.h>

/*
 * tcp 引擎的内部实现， 主要形式是一个死循环的工作线程
 * 它不断的从底层udp套接字和 user_socket_pair 收发包，沟通用户态程序和对端的tcp通信者
 * 它在与对端tcp通信的时候，采取tcp协议的拥塞控制、窗口控制等算法
 * 
 * 这个头文件里声明的函数，应用开发者都不需要关注的。是引擎实现用到的函数
 */



//服务端收到建联请求1
struct btcp_tcpconn_handler *  btcp_handle_sync_rcvd1(char * bigbuffer, 
            struct btcp_tcpsrv_handler* srv, const struct sockaddr_in * client_addr);
//服务端收到建联请求2
int btcp_handle_sync_rcvd2(char * bigbuffer,  struct btcp_tcpconn_handler * handler, 
                        const struct sockaddr_in * client_addr);

//建联后收到数据包，可能没有用户数据，只是ACK或者FIN。c/s两端都会用到                      
int btcp_handle_data_rcvd(char * bigbuffer, int pkg_len, struct btcp_tcpconn_handler * handler, 
            const struct sockaddr_in * client_addr);  

// 客户端发送建联请求后收到服务端应答的处理
int btcp_handle_sync_sent(char * bigbuffer,  struct btcp_tcpconn_handler * handler);


// 检查计时器，看发送报文是否有超时
int btcp_check_send_timeout(struct btcp_tcpconn_handler *handler);

// 尝试发送 send buffer 里的数据
int btcp_try_send(struct btcp_tcpconn_handler *handler);

// 保活。由于服务器和客户端的实现有点差异，所以有个is_server参数来区分
int btcp_keep_alive(struct btcp_tcpconn_handler *handler, char *bigbuffer, bool is_server);

// 收到了tcp数据，上抛给用户，实际上就是写入了  user_socket_pair
int btcp_throw_data_to_user(struct btcp_tcpconn_handler * handler);

// 初始化 tcp连接的资源，例如定时器、发送缓冲区等
int btcp_init_tcpconn(struct btcp_tcpconn_handler *handler);

// 销毁tcp连接 的相关资源，例如定时器、发送缓冲区等
int btcp_destroy_tcpconn(struct btcp_tcpconn_handler *handler, bool is_server);


//发送报文后，要设定定时器。该函数返回一个整数，用作超时时间长度，单位是秒
int btcp_get_timeout_sec(struct btcp_tcpconn_handler *handler);

// 增大拥塞窗口的大小
int btcp_increase_cong_wnd(struct btcp_tcpconn_handler *handler);

// 发生了丢包，缩小拥塞窗口的大小
int btcp_shrink_cong_wnd(struct btcp_tcpconn_handler *handler, bool quick);

// 下面挥手相关的函数：

// 主动挥手者 发送 fin 请求报文，进入 fin_wait1 状态
int btcp_enter_fin_wait1(struct btcp_tcpconn_handler *handler, 
                    char *bigbuffer);

// 被动挥手者 发送 fin 请求报文，进入 last_ack 状态
int btcp_enter_last_ack(struct btcp_tcpconn_handler *handler, 
                    char *bigbuffer);

// 上面两个函数调用的实际的fin 请求发送函数
int btcp_send_fin_request(struct btcp_tcpconn_handler *handler, 
                    char *bigbuffer, 
                    uint32_t * ack_seq);

// 被动挥手者 收到 fin 请求报文后进行ack，进入 close_wait 状态
int btcp_enter_close_wait(struct btcp_tcpconn_handler *handler, 
        union btcp_tcphdr_with_option *tcphdr, 
        char *bigbuffer);
// 主动挥手者 收到 fin 请求报文后进行ack，进入 time_wait 状态
int btcp_enter_time_wait(struct btcp_tcpconn_handler *handler, 
                        union btcp_tcphdr_with_option *tcphdr, 
                        char *bigbuffer);

// 上面两个函数调用的实际的fin ack发送函数
int btcp_send_fin_response(struct btcp_tcpconn_handler *handler, 
                        union btcp_tcphdr_with_option *tcphdr, 
                        char *bigbuffer);






#endif