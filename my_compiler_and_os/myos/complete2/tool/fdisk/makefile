CFLAGS 		=  -g2    -I../../lib
CC			= gcc

OBJS		= buffer.o inode.o super.o namei.o truncate.o hd.o main.o

all: fdisk
	
fdisk:	$(OBJS)
	$(CC) -g2 -o$@ $^ ../../obj/libcommon.a
	./fdisk

%.o : %.c
	$(CC) -c $*.c $(CFLAGS)

clean:
	rm -rf ./*.o
