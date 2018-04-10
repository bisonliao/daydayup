// P1001.cpp : main project file.
// ACM的一道题：
// This problem requires that you write a program to compute the exact value of Rn where R is a real number ( 0.0 < R < 99.999 )
//  and n is an integer such that 0 < n <= 25.
//
//  http://poj.org/problem?id=1001

#include "stdafx.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

//using namespace System;

#define MAX_BYTE_NUM (100)

typedef struct
{
	uint8_t v[MAX_BYTE_NUM];
	int cnt;
}BigInt;

static int bigint_init(BigInt * r)//初始化为0
{
	if (r == NULL) {return -1;}
	int i;
	for (i = 0; i < MAX_BYTE_NUM; ++i)
	{
		r->v[i] = 0;
	}
	r->cnt = 0;
	return 0;
}
static int bigint_init(BigInt * r, uint64_t v)//初始化r为v
{
	if (r == NULL) {return -1;}
	int i;
	for (i = 0; i < MAX_BYTE_NUM; ++i)
	{
		r->v[i] = 0;
	}
	r->cnt = 0;
	for (i = 0; i < 8; ++i)
	{
		r->v[i] = (v >> (8*i)) & 0xff;
	}
	r->cnt = 8;
	return 0;
}
static int bigint_init2(BigInt * r, uint32_t v)//初始化r为v
{
	if (r == NULL) {return -1;}
	int i;
	for (i = 0; i < MAX_BYTE_NUM; ++i)
	{
		r->v[i] = 0;
	}
	r->cnt = 0;
	for (i = 0; i < 4; ++i)
	{
		r->v[i] = (v >> (8*i)) & 0xff;
	}
	r->cnt = 4;
	return 0;
}
//加法，a+b=result
static int bigint_add(const BigInt* a, const BigInt * b, BigInt * result)
{
	if (a == NULL || b == NULL || result == NULL) { return -1;}

	bigint_init(result);
	int m = a->cnt;
	if (b->cnt > m) { m = b->cnt;};
	
	int i;
	uint8_t carry = 0;
	for (i = 0; i < m; ++i)
	{
		uint16_t sum = a->v[i] + b->v[i]+carry;
		if (sum > UINT8_MAX)
		{
			sum = sum & 0x00ff;
			carry = 1;
		}
		else
		{
			carry = 0;
		}
		result->v[i] = sum;
		result->cnt = i+1;
	}
	if (carry&& i < MAX_BYTE_NUM)
	{
		result->v[i] = 1;
		result->cnt = i+1;
	}
	return 0;
}
//乘法： a*b=result
static int bigint_multi(const BigInt* a, uint8_t b, BigInt * result)
{
	if (a == NULL  || result == NULL) { return -1;}

	bigint_init(result);

	int i;
	uint8_t carry = 0;

	for (i = 0; i < a->cnt; ++i)
	{
		uint16_t p = a->v[i] * b + carry;
		if (p > UINT8_MAX)
		{
			carry = (p >> 8) & 0x00ff;

			result->v[i] = p & 0x00ff;
			result->cnt = i+1;
		}
		else
		{
			carry = 0;

			result->v[i] = p;
			result->cnt = i+1;
		}
	}
	if (carry && i < MAX_BYTE_NUM)
	{
		result->v[i] = carry;
		result->cnt = i+1;
	}
	
	return 0;
}
// r左移byte_num个字节
static int bigint_lshift(BigInt * r, uint8_t byte_num)
{
	if (r == NULL) { return -1;}

	int i = r->cnt-1 + byte_num;
	if (i >= MAX_BYTE_NUM) { i = MAX_BYTE_NUM - 1;}
	int j = i - byte_num;
	if (j < 0)
	{
		bigint_init(r);
		return 0;
	}
	for (; j >=0; j--, i--)
	{
		r->v[i] = r->v[j];
	}
	for (; i >=0; i--)
	{
		r->v[i] = 0;
	}
	r->cnt = r->cnt + byte_num;
	if (r->cnt > MAX_BYTE_NUM)
	{
		r->cnt = MAX_BYTE_NUM;
	}
	return 0;
}
//对于比较小的整数a，以uint64_t的方式输出
static uint64_t bigint_getint(const BigInt* a)
{
	if (a == NULL || a->cnt > 8) { return UINT64_MAX;}
	uint64_t vv = 0;
	int i;
	for (i = 0; i< a->cnt; ++i)
	{
		vv = ((uint64_t)(a->v[i]) << (i*8)) + vv;
	}
	return vv;
}
//乘法
static int bigint_multi2(const BigInt* a, uint32_t b, BigInt * result)
{
	if (a == NULL || result == NULL) {return -1;}
	bigint_init(result);

	int i;
	for (i = 0; i < 4; ++i)
	{
		uint8_t c = (b >> (i*8))& 0x0ff;
		BigInt tmp,tmp2;
		bigint_multi(a, c, &tmp);
		bigint_lshift(&tmp, i);
		bigint_add(result, &tmp, &tmp2);
		*result = tmp2;
	}
	return 0;

}
//比较两个数大小，返回 1 0 -1表示a > b, a==b, a<b
static int bigint_compare(const BigInt * a, const BigInt * b)
{
	if (a == NULL || b == NULL ) { return 0;}
	int m = a->cnt;
	if (b->cnt > m) { m = b->cnt;}
	int i;
	for (i = m; i >=0; i--)
	{
		if (a->v[i] > b->v[i])
		{
			return 1;
		}
		if (a->v[i] < b->v[i])
		{
			return -1;
		}
	}
	return 0;
}
//减法a-b=result
static int bigint_sub(const BigInt* a, const BigInt * b, BigInt * result)
{
	if (a == NULL || b == NULL || result == NULL) { return -1;}
	if (bigint_compare(a,b) < 0) { return -2;}

	bigint_init(result);
	int m = a->cnt;
	if (b->cnt > m) { m = b->cnt;};
	
	int i;
	uint8_t carry = 0;
	for (i = 0; i < m; ++i)
	{
		if (a->v[i] >= (b->v[i]+carry))
		{
			result->v[i] = a->v[i] - b->v[i] - carry;
			result->cnt = i+1;

			carry = 0;
		}
		else
		{
			result->v[i] = 256 + a->v[i] - b->v[i] - carry;
			result->cnt = i+1;

			carry = 1;
		}
		
	}
	if (carry)
	{
		return -3;
	}
	return 0;
}
//除法： a/b=result，余数left，效率比较低，一个一个减除数的方式
static int bigint_div(const BigInt * a, const BigInt * b, BigInt * result, BigInt * left)
{
	if (a == NULL || b == NULL || result == NULL || left == NULL) { return -1;}
	
	bigint_init(result);
	BigInt one, aa, zero;
	bigint_init(&one, 1);
	bigint_init(&zero);

	if (bigint_compare(b, &zero) == 0) { return -2;}

	aa = *a;

	while (bigint_compare(&aa, b) >= 0)
	{
		// result += 1;
		BigInt sum, diff;
		bigint_add(result, &one, &sum);
		*result = sum;

		// aa = aa - b;
		bigint_sub(&aa, b, &diff);
		aa = diff;

	
	}
	*left = aa;
	return 0;
}
//除法，相对上面的函数来说效率更高
static int bigint_div(const BigInt * a, uint32_t b, BigInt * result, BigInt * left)
{
	if (a == NULL ||  result == NULL || left == NULL) { return -1;}
	if (b == 0 ) { return -2;}
	bigint_init(result);
	BigInt bb, aa;
	aa = *a;
	bigint_init(&bb, b);

	while (bigint_compare(&aa, &bb) >= 0)
	{
	
		while (aa.cnt > 0 && aa.v[aa.cnt-1] == 0) //cnt找到第一个非0的字节
		{
			aa.cnt--;
		}
		if (aa.cnt == 0) { *left = aa; return 0;} // 被除数是0

		int  len;
	
	
		uint64_t d = 0;
		
		for (len = 1; len <=5 && aa.cnt >= len; len++)
		{
			d = (d << 8) + aa.v[aa.cnt-len];
			
			if (d>=b)
			{
				uint32_t r = d / b;
				uint32_t l = d - b * r;

				//处理余数，更新到aa里
				int i;
				for (i = 0; i < len && i < 4; ++i)
				{
					aa.v[aa.cnt-len+i] = (l >> (i*8)) & 0xff;
				}
				//处理商
				for (i = 0; i < len && i < 4; ++i)
				{
					if (result->v[aa.cnt-len+i] == 0)
					{
					    result->v[aa.cnt-len+i] = (r >> (i*8)) & 0xff;
					}
				}

				break;
			}
		}
	}

	*left = aa;
	int i;
	for (i = 0; i < MAX_BYTE_NUM; ++i)
	{
		if (result->v[i])
		{
			result->cnt = i+1;
		}
	}
	return 0;
}
//以10进制可读的字符串输出
static char * bigint_getstr(const BigInt * a)
{
	static char buf[1024];
	buf[0] = 0;
	if (a == NULL) { return buf;}

	BigInt ten, aa, zero;
	bigint_init(&ten, 1000000);
	bigint_init(&zero);
	aa = *a;

	int i = sizeof(buf)-1;
	buf[i] = 0;
	

	while (bigint_compare(&aa, &zero) > 0)
	{
		BigInt result, left;
		bigint_div(&aa, 1000000, &result, &left);

		aa = result;

		uint32_t l = bigint_getint(&left);
		char tmp[7];
		_snprintf(tmp, 7, "%06d", l);

		int k;
		for (k = 5; k >= 0; --k)
		{
			i--;
			buf[i] = tmp[k];
			
		}

	}
	return &buf[i];

}
//以16进制可读字符串输出
static char * bigint_getHex(const BigInt * a)
{
	static char buf[1024];
	buf[0] = 0;
	if (a == NULL) { return buf;}

	int i;
	int index = 0;
	for (i = a->cnt; i >= 0; --i)
	{
		char tmp[3];
		_snprintf(tmp, 3, "%02X", a->v[i]);
		
		buf[index++] = tmp[0];
		buf[index++] = tmp[1];
	}
	buf[index] = 0;

	

	return buf;

}



int main(int argc, char**argv)
{
    //Console::WriteLine(L"Hello World");
	int i;

	BigInt a, b,c, d;
	bigint_init2(&a, 95123);
	//printf("%s\n", bigint_getstr(&a));

	
	
	for ( i = 0; i < 11; ++i)
	{
		bigint_multi2(&a, 95123, &c);
		a = c;
		printf("i=%d\n", i);
	}
	printf("a=%s\n", bigint_getstr(&a));//验证：95.123的12次方，548815620517731830194541.899025343415715973535967221869852721
	
	/*
	bigint_div(&a, 123, &c, &d);
	printf("a=%llu\n", bigint_getint(&d));
	printf("a=%llu\n", bigint_getint(&c));
	*/

	
	
	


	scanf("%d", &i);
    return 0;
}
