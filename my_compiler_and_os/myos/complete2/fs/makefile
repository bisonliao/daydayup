CFLAGS 		=  -O2   -I../lib -I../inc -fno-builtin 
CC			= gcc

OBJS		= buffer.o inode.o super.o namei.o truncate.o

all: ../obj/libfs.a
	
../obj/libfs.a:	$(OBJS)
	ar rv ../obj/libfs.a $(OBJS)

%.o : %.c
	$(CC) -c $*.c $(CFLAGS)

clean:
	rm -rf ../obj/libfs.a; rm -rf ./*.o
