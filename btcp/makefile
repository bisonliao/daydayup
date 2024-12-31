all: cli srv

cli:tcpcli.c common.c circular_queue.c
	gcc -o cli tcpcli.c common.c circular_queue.c -lpthread 
srv:tcpsrv.c common.c circular_queue.c
	gcc -o srv tcpsrv.c common.c circular_queue.c -lpthread 
	scp srv pi@192.168.0.11:~/btcp/

clean: 
	rm *.o cli srv
