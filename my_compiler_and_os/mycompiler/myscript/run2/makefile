#CC			= g++  -g
CC			= g++ -O3

PERL_CFLAGS	= `perl -MExtUtils::Embed -e ccopts `
PERL_LIBS	= `perl -MExtUtils::Embed -e ldopts`

#如果需要让流程支持第归，加上_RECURSIVE_
#如果不支持第归，可获得性能上的提高
#CFLAGS		=  -I/usr/local/mysql/include -D_RECURSIVE_
CFLAGS		=  -I/usr/local/mysql/include 

LNKFLG		= $(PERL_LIBS) -L/usr/local/mysql/lib -lmysqlclient  -lm  -lz

OBJS		= mem.o var.o script.o run.o filefunc.o util.o stringfunc.o perlc.o ipcfunc.o mysqlfunc.o datetimefunc.o \
				InitFuncMap.o mapfunc.o
 
all: myrun liblilyrun.a

install:liblilyrun.a
	su -c 'cp liblilyrun.a /usr/lib ; if [ !  -d /usr/include/lily  ] ; then mkdir /usr/include/lily;  chmod a+x /usr/include/lily; fi; cp *.h /usr/include/lily'
	

myrun:$(OBJS) main.o
	$(CC) -o myrun main.o $(OBJS) $(LNKFLG)
liblilyrun.a: $(OBJS)
	ar -rv $@ $^
	
perlc.o: perlc.cpp 
	$(CC) -c -o$@ $^  $(CFLAGS) $(PERL_CFLAGS)

%.o:%.cpp 
	$(CC) -c -o $@ $^ $(CFLAGS)



clean:
	-@rm *.o myrun librun.a
