#include "BigDecimal.h"

extern "C" {
#include <string.h>
}


int bigdecimal_assign(BigDecimal* a, const char * str)
{
	if (a == NULL )
	{
		return -1;
	}
	memset(a, 0, sizeof(BigDecimal));
	if (str == NULL || strlen(str) == 0)
	{
		return 0;
	}
	const char * point = strchr(str, '.');
	int len = strlen(str);
	if (point == NULL)
	{
		int j = DIGITS_NUM_BEFORE_POINT - 1;
		int i = len - 1;
		for (; i >= 0 && j >=0; --i, --j)
		{
			if (!(str[i] >= '0' && str[i] <= '9'))
			{
				return -1;
			}
			a->digits[j] = str[i] - '0';
		}
	}
	else
	{
		int j = DIGITS_NUM_BEFORE_POINT - 1;
		int i = point - str-1;
		for (; i >= 0 && j >= 0; --i, --j)
		{
			if (!(str[i] >= '0' && str[i] <= '9'))
			{
				return -1;
			}
			a->digits[j] = str[i] - '0';
		}

		j = DIGITS_NUM_BEFORE_POINT;
		i = point - str+1 ;
		for (; i < len && j < DIGITS_NUM; ++i, ++j)
		{
			if (!(str[i] >= '0' && str[i] <= '9'))
			{
				return -1;
			}
			a->digits[j] = str[i] - '0';
		}

	}
	return 0;

}
int bigdecimal_tostring(const BigDecimal* a, char * str)
{
	int i;
	bool nonzero_appared = false;
	int index = 0;
	for (i = 0; i < DIGITS_NUM_BEFORE_POINT; ++i)
	{
		if (nonzero_appared)
		{
			str[index++] = a->digits[i] + '0';
		}
		else
		{
			if (a->digits[i] == 0)
			{
				continue;
			}
			else
			{
				nonzero_appared = true;
				str[index++] = a->digits[i] + '0';
			}
		}

	}
	nonzero_appared = false;
	str[index++] = '.';
	int save_index = index;
	for (; i < DIGITS_NUM; ++i)
	{
		if (a->digits[i] != 0)
		{
			nonzero_appared = true;
		}
		str[index++] = a->digits[i] + '0';
	}
	str[index++] = 0;
	if (!nonzero_appared)
	{

		str[save_index+1] = 0;
	}
	
	
	return 0;

}

int bigdecimal_multi(const BigDecimal* a, unsigned int b, BigDecimal *result)
{
	if (a == NULL || result == NULL)
	{
		return -1;
	}
	memset(result, 0, sizeof(BigDecimal));
	unsigned int carry = 0;
	int i;
	for (i = DIGITS_NUM - 1; i > 0; --i)
	{
		unsigned int accum = a->digits[i] * b + carry;
		result->digits[i] = accum % 10;
		carry = accum / 10;
	}
	return 0;
}

int bigdecimal_div(const BigDecimal* a, unsigned int b, BigDecimal *result)
{
	if (a == NULL || result == NULL)
	{
		return -1;
	}
	memset(result, 0, sizeof(BigDecimal));
	int i;
	unsigned int remainder = 0;
	for (i = 0; i < DIGITS_NUM; ++i)
	{
		unsigned int dividend = (a->digits[i]+remainder*10) ;
		result->digits[i] = dividend / b;
		remainder = dividend % b;
		
	}
	
	return 0;
}
int bigdecimal_add(const BigDecimal* a, const BigDecimal* b, BigDecimal *result)
{
	if (a == NULL || b == NULL || result == NULL)
	{
		return -1;
	}
	memset(result, 0, sizeof(BigDecimal));
	int i;
	unsigned int carry = 0;
	for (i = DIGITS_NUM-1; i >= 0; --i)
	{
		unsigned int sum = a->digits[i] + b->digits[i] + carry;
		result->digits[i] = sum % 10;
		carry = sum / 10;
	}
	return 0;
}
int bigdecimal_compute_pai()
{
	/*
	BigDecimal a;
	bigdecimal_assign(&a, "12");
	BigDecimal result;
	bigdecimal_multi(&a, 3, &result);

	char str[100];
	bigdecimal_tostring(&result, str);
	printf("%s\n", str);

	bigdecimal_div(&a, 32, &result);
	bigdecimal_tostring(&result, str);
	printf("%s\n", str);
	*/
	BigDecimal result,tmp, part;
	bigdecimal_assign(&result, "1.0");
	bigdecimal_div(&result, 3, &part);

	bigdecimal_add(&result, &part, &tmp);
	result = tmp;

	unsigned int a=2, b=5;
	BigDecimal zero;
	bigdecimal_assign(&zero, "0");

	for (int i = 0; i < 5000000; ++i)
	{
		bigdecimal_multi(&part, a, &tmp);
		bigdecimal_div(&tmp, b, &part);

		bigdecimal_add(&result, &part, &tmp);
		result = tmp;

		a++;
		b += 2;

		if (0 == (i % 10000))
		{
			printf("%d\n", i);
			if (memcmp(&part, &zero, sizeof(BigDecimal)) == 0) // part is as little as zero
			{
				printf("no part added, i=%d!\n", i);
				break;
			}
		}
	}
	bigdecimal_multi(&result, 2, &tmp);
	result = tmp;
	
	char str[DIGITS_NUM * 2];
	bigdecimal_tostring(&result, str);
	printf("%s\n", str);
	return 0;
}
int bigdecimal_compute_e()
{
    BigDecimal e;
    BigDecimal v;
    bigdecimal_assign(&e, "2.5");
    bigdecimal_assign(&v, "0.5");


    for (int i = 3; ; ++i)
    {
        BigDecimal result1, result2;
        bigdecimal_div(&v, i, &result1);
        bigdecimal_add(&e, &result1, &result2);

        BigDecimal oldvalue = e;
        e = result2;
        v = result1;

      
        if (memcmp(&e, &oldvalue, sizeof(BigDecimal)) == 0)
        {
            printf("i=%d\n", i);
            break;
        }
        
    }
    static char str[100000];
    bigdecimal_tostring(&e, str);
    printf("%s\n", str);
    printf("len=%d\n", strlen(str));
    return 0;
}
