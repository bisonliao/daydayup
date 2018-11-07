#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#include <math.h>


int main(int argc, char ** argv)
{
	if (argc < 4)
	{
		printf("%s mean stdev signum\n", argv[0]);
		return -1;
	}

	double mean = atof(argv[1]);
	double stdev = atof(argv[2]);
	int signum = atoi(argv[3]);

	srand( time(NULL) );

	int i, j;
	for (i = 0; i < signum; ++i)
	{
		double  r1 = ((double)rand()) / RAND_MAX;
		double  r2 = ((double)rand()) / RAND_MAX;

		double x = cos(2*3.1415926*r2) * sqrt( -2 * log(r1) );

		x *= stdev;
		x += mean;
		
		printf("%.4f\n", x);
	}
	return 0;

}
