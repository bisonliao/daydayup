CFLAGS=-I.  -I../include
LIBS=-lfl -L../lib -llily
CC=g++

../bin/compile: lily.lex.o lily.yacc.o  Token_Stack.o int_Stack.o \
Labels_Stack.o FunCfg.o FlowName_Stack.o
	${CC} -o $@ $^ ${LIBS}

Token_Stack.o:Token_Stack.cpp
	${CC} -c -o $@ $^ ${CFLAGS}
FlowName_Stack.o:FlowName_Stack.cpp
	${CC} -c -o $@ $^ ${CFLAGS}
int_Stack.o:int_Stack.cpp
	${CC} -c -o $@ $^ ${CFLAGS}
Labels_Stack.o:Labels_Stack.cpp
	${CC} -c -o $@ $^ ${CFLAGS}
lily.lex.o:lily.lex.cpp compile.yacc.h
	${CC} -c -o $@ lily.lex.cpp ${CFLAGS}
lily.yacc.o:lily.yacc.cpp
	${CC} -c -o $@ $^ ${CFLAGS}
lily.lex.cpp:Lily_compile.lex
	lex -o$@ $^
compile.yacc.h lily.yacc.cpp:Lily_compile.yacc
	yacc -d -olily.yacc.cpp $^
	mv lily.yacc.cpp.h compile.yacc.h
FunCfg.o:FunCfg.lex
	flex -oFunCfg.cpp -Pfc  FunCfg.lex
	${CC} -c -o FunCfg.o FunCfg.cpp  -I../include

clean:
	-@rm -rf *.o
