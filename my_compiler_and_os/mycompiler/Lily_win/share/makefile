OBJS=AnsiString.o tool.o MemBlock.o
CFLAGS=-I../include
CC=g++

all: liblily.a

liblily.a: ${OBJS}
	ar -qv $@ ${OBJS}
	cp $@ ../lib/
AnsiString.o:AnsiString.cpp
	${CC} -c -o $@ $^ ${CFLAGS}
tool.o:tool.c
	${CC} -x c++ -c -o $@ $^ ${CFLAGS}
MemBlock.o:MemBlock.cpp
	${CC} -c -o $@ $^ ${CFLAGS}

clean:
	-rm ./*.o
