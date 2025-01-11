CFLAGS	= -g -fsanitize=address -I/usr/include/glib-2.0 -I/usr/lib/arm-linux-gnueabihf/glib-2.0/include
CLIBS	=-lglib-2.0 -lpthread
FILES	=  btcp_engine.c btcp_timeout.c btcp_send_queue.c  btcp_recv_queue.c   btcp_rtt.c \
			btcp_selective_ack_blocklist.c tool.c

all: demo_cli demo_srv libbtcp.a

demo_cli:demo_tcpcli.c libbtcp.a
	gcc -o demo_cli  $^  $(CFLAGS) $(CLIBS)
demo_srv:demo_tcpsrv.c libbtcp.a
	gcc -o demo_srv $^  $(CFLAGS) $(CLIBS)
	scp srv pi@192.168.0.11:~/btcp/


OBJS = $(FILES:.c=.o)
libbtcp.a: $(OBJS)
	ar rcs $@ $^
	rm $(OBJS)

# 编译源文件为目标文件
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
	
clean: 
	-rm *.o demo_cli demo_srv libbtcp.a

# btcp作者单元测试某个模块的时候用的
ttx:btcp_rtt.c tool.c
	gcc -o ttx $^ $(CFLAGS) $(CLIBS)
