CC=g++
CFLAGS=-I../include -I../Lily_run
LIBS=-L../lib -llily

all:librun_fun.so

librun_fun.so:Run_Fun.o ../Lily_run/Variable.o
	${CC} -shared -o $@ $^ ${LIBS}
	cp $@ ../lib/
Run_Fun.o:Run_Fun.cpp
	${CC} -c -o $@ $^ ${CFLAGS}
../Lily_run/Variable.o:../Lily_run/Variable.cpp
	${CC} -c -o $@ $^ ${CFLAGS}
