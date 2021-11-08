CFLAGS 		= -std=c99 -g  -I../skiplist  -D_BSD_SOURCE

all: ../skiplist/skiplist.o main.o MemTable.o SSTable.o
	gcc -o main.exe main.o MemTable.o SSTable.o ../skiplist/skiplist.o  -lpthread  

MemTable.o:MemTable.c
	gcc -c $(CFLAGS) -o MemTable.o MemTable.c 

main.o:main.c
	gcc -c $(CFLAGS) -o main.o main.c 

SSTable.o:SSTable.c
	gcc -c $(CFLAGS) -o SSTable.o SSTable.c 