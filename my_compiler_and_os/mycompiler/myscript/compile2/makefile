#CC			= g++ -g -pg
CC			= g++ -O2
OBJ			= lex.yy.o token.o  expr.o parse.o  compile.o
CFLAGS		= 

all: mylex   liblilycompile.a

install:liblilycompile.a compile.h  token.h
	su -c 'cp liblilycompile.a /usr/lib ; if [ !  -d /usr/include/lily  ] ; then mkdir /usr/include/lily;  chmod a+x /usr/include/lily; fi; cp *.h /usr/include/lily'

mylex: $(OBJ) main.o
	$(CC) -o mylex $(OBJ) main.o
liblilycompile.a: $(OBJ)
	ar -rv $@ $^

lex.yy.o: lex.yy.c
	$(CC) -c -x c++ lex.yy.c -o lex.yy.o $(CFLAGS)
lex.yy.c:lily.lex
	flex lily.lex

%.o:%.cpp
	$(CC) -c $*.cpp -o $*.o $(CFLAGS)


clean:
	-rm *.o;touch *.lex; rm mylex libcompile.a
