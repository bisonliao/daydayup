CFLAGS 		= -g

ttx:tdigest2.o centroid.o
	gcc $(CFLAGS) -o $@ $^ 

tdigest2.o:tdigest2.c
	gcc $(CFLAGS) -c -o $@ $^

centroid.o:centroid.c
	gcc $(CFLAGS) -c -o $@ $^
