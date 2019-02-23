CC			= g++ -g
CFLAGS		=
LNKFLG		=

OBJS		= mem.o var.o script.o main.o

all: myrun

myrun:$(OBJS)
	$(CC) -o myrun $(OBJS)
	

mem.o:mem.cpp mem.h
	$(CC) -c -o mem.o mem.cpp

var.o:var.cpp var.h
	$(CC) -c -o var.o var.cpp

script.o:script.cpp script.h
	$(CC) -c -o script.o script.cpp

main.o:main.cpp 
	$(CC) -c -o main.o main.cpp


clean:
	-@rm *.o
