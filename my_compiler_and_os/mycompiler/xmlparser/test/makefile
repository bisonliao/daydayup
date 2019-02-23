all:tt

tt:test.o ../bin/libxmlparser.a
	g++ -o tt test.o ../bin/libxmlparser.a
test.o:test.cpp
	g++ -c -o test.o test.cpp -I../inc

