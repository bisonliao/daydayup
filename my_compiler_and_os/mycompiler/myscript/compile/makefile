CC			= g++ -g

all: mylex

mylex:lex.yy.o token.o main.o expr.o parse.o
	$(CC) -o mylex lex.yy.o token.o  main.o expr.o parse.o

lex.yy.o: lex.yy.c
	$(CC) -c -x c++ lex.yy.c -o lex.yy.o
lex.yy.c:lily.lex
	lex lily.lex

token.o:token.cpp
	$(CC) -c token.cpp -o token.o
main.o:main.cpp
	$(CC) -c main.cpp -o main.o
expr.o:expr.cpp
	$(CC) -c expr.cpp -o expr.o
parse.o:parse.cpp
	$(CC) -c parse.cpp -o parse.o


clean:
	-rm *.o;touch *.lex
