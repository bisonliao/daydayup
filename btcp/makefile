CFLAGS	= -g -fsanitize=address -I/usr/include/glib-2.0 -I/usr/lib/arm-linux-gnueabihf/glib-2.0/include
CLIBS	=-lglib-2.0 -lpthread
FILES	= tool.c btcp_engine.c btcp_timeout.c btcp_send_queue.c  btcp_recv_queue.c btcp_range.c  btcp_rtt.c

all: cli srv

cli:tcpcli.c $(FILES)
	gcc -o cli  $^  $(CFLAGS) $(CLIBS)
srv:tcpsrv.c $(FILES)
	gcc -o srv $^  $(CFLAGS) $(CLIBS)
	scp srv pi@192.168.0.11:~/btcp/

clean: 
	rm *.o cli srv

ttx:btcp_rtt.c tool.c
	gcc -o ttx $^ $(CFLAGS) $(CLIBS)
