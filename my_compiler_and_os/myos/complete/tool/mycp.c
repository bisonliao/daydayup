#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>


char buf[2000000];
int main(int argc, char ** argv)
{
	int fdr, fdw;
	int len;
	if (argc < 5)
	{
		return 0;
	}
	fdw = open(argv[1], O_WRONLY);
	if (fdw < 0)
	{
		perror("open:");
		return -1;
	}
	fdr = open(argv[2], O_RDONLY);
	if (fdr < 0)
	{
		perror("open:");
		return -1;
	}
	len = read(fdr, buf, sizeof(buf));
	if (len < 0)
	{
		perror("read:");
		return -1;
	}
	if (len > 512)
	{
		fprintf(stderr, "TOO BIG!\n");
		return -1;
	}
	if (write(fdw, buf, len) != len)
	{
		perror("write");
		return -1;
	}
	printf("transfer %d bytes\n", len);

	close(fdr);
	fdr = open(argv[3], O_RDONLY);
	if (fdr < 0)
	{
		perror("open:");
		return -1;
	}
	lseek(fdw, 512, SEEK_SET);
	len = read(fdr, buf, sizeof(buf));
	if (len > 2048)
	{
		fprintf(stderr, "too long!\n");
		return -1;
	}
	if (len < 0)
	{
		perror("read:");
		return -1;
	}
	if (write(fdw, buf, len) != len)
	{
		perror("write");
		return -1;
	}
	printf("transfer %d bytes\n", len);
	close(fdr);
	fdr = open(argv[4], O_RDONLY);
	if (fdr < 0)
	{
		perror("open:");
		return -1;
	}
	lseek(fdw, 512*5, SEEK_SET);
	len = read(fdr, buf, sizeof(buf));
	if (len < 0)
	{
		perror("read:");
		return -1;
	}
	if (write(fdw, buf, len) != len)
	{
		perror("write");
		return -1;
	}
	printf("transfer %d bytes\n", len);
	return 0;
}
