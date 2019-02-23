CC=g++
#CFLAGS=-I. -I../include -D_DEBUG
CFLAGS=-I. -I../include
LIBS=-L../share -llily -L../Run_Fun -lrun_fun

OBJS=LabelIndex_List.o Variable.o instruct_list.o Variable_Stack.o int_Stack.o \
run_main.o 

all:lily_run 

lily_run:${OBJS}
	${CC} -o $@ $^ ${LIBS}
	mv lily_run ../bin

LabelIndex_List.o:LabelIndex_List.cpp
	${CC} -c -o $@ $^ ${CFLAGS}
Variable.o:Variable.cpp
	${CC} -c -o $@ $^ ${CFLAGS}
instruct_list.o:instruct_list.cpp
	${CC} -c -o $@ $^ ${CFLAGS}
Variable_Stack.o:Variable_Stack.cpp
	${CC} -c -o $@ $^ ${CFLAGS}
int_Stack.o:int_Stack.cpp
	${CC} -c -o $@ $^ ${CFLAGS}
run_main.o:run_main.cpp
	${CC} -c -o $@ $^ ${CFLAGS}

clean:
	-rm ./*.o
