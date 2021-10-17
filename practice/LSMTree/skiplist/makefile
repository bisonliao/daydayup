CFLAGS =  -std=c99 -g

all: skiplist.o main.o
	gcc -o main.exe main.o skiplist.o

skiplist.o:skiplist.c
	gcc -c $(CFLAGS) -o skiplist.o skiplist.c

main.o:main.c
	gcc -c $(CFLAGS) -o main.o main.c