CC=g++
CFLAGS=-I../include -I../Lily_Run
LIBS=-L../lib -llily

all:librun_fun.so

librun_fun.so:Run_Fun.o ../Lily_Run/Variable.o
	${CC} -shared -o $@ $^ ${LIBS}
	cp $@ ../lib/
Run_Fun.o:Run_Fun.cpp
	${CC} -c -o $@ $^ ${CFLAGS}
../Lily_Run/Variable.o:../Lily_Run/Variable.cpp
	${CC} -c -o $@ $^ ${CFLAGS}

clean:
	-rm ./*.o
