#pragma once

/*
 * calculate PI
 * algorithm : Pi/2 = 1 + 1/3 + 1/3 * 2/5 + 1/3 * 2/5 * 3/7 + Part_n * [(n+1) / (2n+1)] + ...
 * this algorithm can NOT be executed parallelly, so I can NOT implement it with GPU.
 * 
 * use BigDecimal defined by myself to calculate.
 */

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

