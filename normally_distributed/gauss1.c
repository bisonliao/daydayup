#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>


int main(int argc, char ** argv)
{
	if (argc < 4)
	{
		printf("%s mean stdev signum\n", argv[0]);
		return -1;
	}
	//For each sample in the signal: 
	//(1) add twelve random numbers, 
	//(2) subtract six to make the mean equal to zero, 
	//(3) multiply by the standard deviation desired, and 
	//(4) add the desired mean.

	double mean = atof(argv[1]);
	double stdev = atof(argv[2]);
	int signum = atoi(argv[3]);

	srand( time(NULL) );

	int i, j;
	for (i = 0; i < signum; ++i)
	{
		double sum = 0;
		for (j = 0; j < 12; ++j)
		{
			double  v = ((double)rand()) / RAND_MAX;
			sum += v;
		}
		sum -= 6.0;
		sum *= stdev;
		sum += mean;

		printf("%.4f\n", sum);
	}
	return 0;

}
