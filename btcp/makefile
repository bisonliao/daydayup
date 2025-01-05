CFLAGS	=-I/usr/include/glib-2.0 -I/usr/lib/arm-linux-gnueabihf/glib-2.0/include
CLIBS	=-lglib-2.0 -lpthread

all: cli srv

cli:tcpcli.c common.c circular_queue.c
	gcc -o cli tcpcli.c common.c circular_queue.c -lpthread 
srv:tcpsrv.c common.c circular_queue.c
	gcc -o srv tcpsrv.c common.c circular_queue.c -lpthread 
	scp srv pi@192.168.0.11:~/btcp/

clean: 
	rm *.o cli srv

ttx:btcp_range.c
	gcc -o ttx btcp_range.c $(CFLAGS) $(CLIBS)
