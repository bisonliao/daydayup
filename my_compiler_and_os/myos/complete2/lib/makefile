CFLAGS 		=   -I../inc -fno-builtin 
CC			= gcc

OBJS		= common_func.o cycle_buf.o hash.o 

all: ../obj/libcommon.a ../obj/redefine.o
	
../obj/libcommon.a:	$(OBJS)
	ar rv ../obj/libcommon.a $(OBJS)
../obj/redefine.o:redefine.c
	$(CC) -o$@ -c redefine.c $(CFLAGS)


%.o : %.c
	$(CC) -c $*.c $(CFLAGS)
%.o : %.cpp
	g++ -c $*.cpp $(CFLAGS)

clean:
	rm -rf ../obj/libcommon.a; rm -rf ./*.o
