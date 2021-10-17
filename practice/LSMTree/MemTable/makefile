CFLAGS 		= -std=c99 -g  -I../skiplist 

all: ../skiplist/skiplist.o main.o MemTable.o
	gcc -o main.exe main.o MemTable.o ../skiplist/skiplist.o  -lpthread  

MemTable.o:MemTable.c
	gcc -c $(CFLAGS) -o MemTable.o MemTable.c 

main.o:main.c
	gcc -c $(CFLAGS) -o main.o main.c 