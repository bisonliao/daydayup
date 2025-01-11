#ifndef __BTCP_API_H_INCLUDED__
#define __BTCP_API_H_INCLUDED__

#include "btcp.h"

/*
 * 暴露给开发者使用引擎的几个主要函数
 */

#define MAX_CONN_ALLOWED (10000) //tcp srv单进程最多处理多少个tcp连接

// tcp server侧，用户程序通过调用 btcp_tcpsrv_listen 开始监听
int btcp_tcpsrv_listen(const char * ip, short int port, struct btcp_tcpsrv_handler * srv);

// tcp client侧，用户程序通过调用 btcp_tcpcli_connect 建立连接
int btcp_tcpcli_connect(const char * ip, short int port, struct btcp_tcpconn_handler * handler);

// tcp client侧，用户程序通过调用 btcp_tcpcli_new_loop_thread 创建tcp内核引擎工作线程
// 用户程序 则通过读写 handler->user_socket_pair[0]套接字实现tcp的传输
int btcp_tcpcli_new_loop_thread(struct btcp_tcpconn_handler *handler); 

// tcp server侧，用户程序通过调用 btcp_tcpsrv_new_loop_thread 创建tcp内核引擎工作线程
// 用户程序 则通过读写 各个连接对应的 handler->user_socket_pair[0] 套接字实现tcp的传输
int btcp_tcpsrv_new_loop_thread(struct btcp_tcpsrv_handler * srv);

//tcp server侧，用户程序通过调用 btcp_tcpsrv_get_all_conn_fds 获取服务器所有的 btcp_tcpconn_handler
// 返回 int fd的列表。 status可以为NULL，不为NULL的时候，指定只返回处于该status状态的连接
// 因为 连接可能被tcp引擎不断的创建和销毁，那么要注意：
// 1) 这个列表用一次之后就释放，重新获取
// 2) 读写fd的过程中注意做一些容错。例如fd可能已经关闭了
GList *  btcp_tcpsrv_get_all_conn_fds(struct btcp_tcpsrv_handler * srv, int * status);

//释放保存了的 int fd 的GList，也会释放每个元素
void btcp_free_conns_in_glist(GList * conns); 

#endif