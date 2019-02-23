CC			= g++ -g 
#CC			= g++ -O2


PERL_LIBS   = `perl -MExtUtils::Embed -e ldopts`
MYSQL_LIBS 	= -L/usr/local/mysql/lib -lmysqlclient  -lm  -lz
CFLAGS		= -I/usr/include/lily 
LNKFLG		=   -llilyprep -llilycompile -llilyrun $(PERL_LIBS) $(MYSQL_LIBS)

OBJS		= 
 
all: bisonliao

install:bisonliao
	su -c 'cp bisonliao /usr/bin; cp lilyall.h /usr/include'

bisonliao:$(OBJS) main.o
	$(CC) -o $@ main.o $(OBJS) $(LNKFLG)

%.o:%.cpp 
	$(CC) -c -o $@ $^ $(CFLAGS)



clean:
	-@rm *.o bisonliao
