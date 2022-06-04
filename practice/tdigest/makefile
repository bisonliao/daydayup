CFLAGS 		= -pg

ttx:tdigest.o centroid.o
	gcc $(CFLAGS) -o $@ $^ 

tdigest.o:tdigest.c
	gcc $(CFLAGS) -c -o $@ $^

centroid.o:centroid.c
	gcc $(CFLAGS) -c -o $@ $^