#include "global.h"

int int_to_hex(uint32_t v, char * buf, size_t bufsz)
{
	uint8_t tmp;
	int i, j;

	if (bufsz < 9)
	{
		return -1;	
	}
	buf[8] = '\0';

	for (i = 7, j = 0; i >= 0; --i, ++j)
	{
		tmp = (v >> (4*i)) & 0xf;
		if (tmp < 10)
		{
			tmp += '0';
		}
		else
		{
			tmp = tmp - 10 + 'A';
		}
		buf[j] = tmp;
	}
	return 0;
}


void div_uint64(uint64_t a, uint64_t b, uint64_t * res)
{
	*res = 0;
	while (a > b)
	{
		a -= b;
		++(*res);
	}
}
