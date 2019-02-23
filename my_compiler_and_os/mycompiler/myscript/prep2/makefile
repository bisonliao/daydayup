#CC		= g++ -g -pg
CC		= g++ -O2

CFLAGS	= -I../run2
LIBS	=
OBJS	= prep.o

all: liblilyprep.a

install:liblilyprep.a
	su -c 'cp liblilyprep.a /usr/lib ; if [ !  -d /usr/include/lily  ] ; then mkdir /usr/include/lily;  chmod a+x /usr/include/lily; fi; cp *.h /usr/include/lily'

liblilyprep.a: $(OBJS)
	ar -rv $@ $^

prep.cpp : prep.lex
	flex -o$@ -Pprep_ $^ 

%.o : %.cpp
	$(CC) -c -o$@ $^ $(CFLAGS)


clean:
	-@rm *.o bl_prep
	-@touch prep.lex
