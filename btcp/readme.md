这个模块尝试用udp在用户态实现tcp协议。
这个模块叫做btcp，是bison tcp的缩写。（有点不自量力）
makefile可以编译出 libbtcp.a库，应用开发者可以使用这个库。
开发者需要引用的头文件只有btcp_api.h，需要关注的函数也都在btcp_api.h里。
关于如何使用btcp库，开发者可以参考demo_tcpcli.c和demo_tcpsrv.c的代码。

btcp库主要依赖 libglib库和pthread库。前者提供了纯c的容器（list/hash table等）和日志功能
btcp会在用户进程中创建引擎工作线程，它不断的从底层udp套接字和 user_socket_pair 收发包，沟通用户态程序和对端的tcp通信者。
引擎工作线程采取拥塞控制、窗口控制等算法模拟实现tcp协议。

bison 2025年一月份于广州南沙。

