#pragma once

/*
 * calculate PI
 * algorithm : Pi/2 = 1 + 1/3 + 1/3 * 2/5 + 1/3 * 2/5 * 3/7 + Part_n * [(n+1) / (2n+1)] + ...
 * this algorithm can NOT be executed parallelly, so I can NOT implement it with GPU.
 * 
 * calculate E
 * algorithm: e=1+1+1/2+1/3!+…+1/n!+...
 * BigDecimal is useful for it.
 * 
 * use BigDecimal defined by myself to calculate.
 */
/**********************************************************
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
************************************************************/

#include <stdlib.h>
#include <stdio.h>

#define DIGITS_NUM_AFTER_POINT (10000)
#define DIGITS_NUM_BEFORE_POINT (20)

#define DIGITS_NUM (DIGITS_NUM_AFTER_POINT+DIGITS_NUM_BEFORE_POINT)

typedef struct
{
	unsigned char digits[DIGITS_NUM];

} BigDecimal;

int bigdecimal_assign(BigDecimal* a, const char * str);
int bigdecimal_tostring(const BigDecimal* a, char * str);
int bigdecimal_multi(const BigDecimal* a, unsigned int b, BigDecimal *result);
int bigdecimal_div(const BigDecimal* a, unsigned int b, BigDecimal *result);
int bigdecimal_add(const BigDecimal* a, const BigDecimal* b, BigDecimal *result);

int bigdecimal_compute_pai();

