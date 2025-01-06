#ifndef BTCP_ENGINE_H_INCLUDED
#define BTCP_ENGINE_H_INCLUDED

#include "btcp.h"
#include "btcp_timeout.h"
#include "btcp_range.h"
#include "btcp_send_queue.h"
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <glib.h>
#include <string.h>
#include <errno.h>

/*
该文档定义引擎内部使用的一些函数和符号，不暴露给开发者
*/

#if 0


//服务端收到建联请求1
int btcp_handle_sync_rcvd1(char * bigbuffer,   
                            struct btcp_tcpsrv_handler* srv, 
                            const struct sockaddr_in * client_addr);
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

// 尝试发送 发送缓冲区里的数据
int btcp_try_send(struct btcp_tcpconn_handler *handler);
#endif




#endif