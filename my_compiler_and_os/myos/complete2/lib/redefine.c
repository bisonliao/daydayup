#include "global.h"
#include "redefine.h"

size_t strlen(const char *s)
{
	size_t ret = 0;
	if (s == NULL)
	{
		return 0;
	}
	while (s[ret] != '\0')
	{
		++ret;
	}
	return ret;
}
int strncmp(const char * a, const char *b, size_t len)
{
    size_t i = 0;
    if (a == NULL || b == NULL) {return -2;}

    if (len == 0) { return 0;}

    for (i = 0; i < len; ++i)
    {
        if (a[i] == b[i])
        {
            continue;
        }
        return (a[i] > b[i] ? 1:-1);
    }
    return 0;
}

void memset(void * p, uint8_t v, uint32_t sz)
{
	while (sz > 0)
	{
		*((uint8_t*)p+sz-1) = v;
		--sz;
	}
}
void strcpy(char * dst, const char * src)
{
	int i = 0;
	while (src[i])
	{
		dst[i] = src[i];
		++i;
	}
	dst[i] = '\0';
}
void strncpy(char * dst, const char * src, uint32_t sz)
{
	int i = 0;
    if (sz == 0) { return;}

	while (src[i] && i < (sz-1))
	{
		dst[i] = src[i];
		++i;
	}
	dst[i] = '\0';
}
void memcpy(void * dst, const void * src, uint32_t sz)
{
	uint32_t i = 0;
	if (sz > 4)
	{
		for (i = 0; i < (sz-4); i = i + 4) /*每次传输四字节,提高效率*/
		{
			*((uint32_t*)(dst+i)) = *((uint32_t*)(src+i));
		}
	}
	for (; i < (sz); ++i)
	{
		*((uint8_t*)dst+i) = *((uint8_t*)src+i);
	}
}
int memcmp(const void * a, const void * b, size_t sz)
{
	const unsigned char * p1, *p2;
	size_t i;
	p1 = (const char *)a;
	p2 = (const char *)b;
	for (i = 0; i < sz; ++i)
	{
		if (p1[i] > p2[i])
		{
			return 1;
		}
		if (p1[i] > p2[i])
		{
			return -1;
		}
	}
	return 0;
}


int snprintf(char * buf, size_t bufsz, const char *fmt, ...)
{
	int len;
	va_list va;
	va_start(va, fmt);
	len = vsprintf(buf, bufsz, fmt, va);
	if (len < sizeof(buf))
	{
		buf[len] = '\0';
	}
	return len;
}
int printf(const char *fmt, ...)
{
	int len;
	va_list va;
    char buf[512];

	va_start(va, fmt);
	len = vsprintf(buf, sizeof(buf), fmt, va);
	if (len < sizeof(buf))
	{
		buf[len] = '\0';
	}
    _cout(buf, len);
    return len;
}
long int strtol(const char *nptr, char **endptr, int base)
{
    int i = 0;
    long int result = 0;
    int minus_flag = 1;
    int scan_number_flag = 0;
    int invalid_char_flag = 0;
    if (nptr == NULL || base < 2 || base > 16)
    {
        return -1;
    }
    for (i = 0; *(nptr+i) != '\0'; ++i)
    {
        char c = *(nptr+i);
        int v = 0;
        if (c == '+')
        {
            if (scan_number_flag)
            {
                invalid_char_flag = 1;
                break;
            }
            continue;
        }
        else if ( c == '-')
        {
            if (scan_number_flag)
            {
                invalid_char_flag = 1;
                break;
            }
            minus_flag = -minus_flag;
            continue;
        }
        else if  ( c >= '0' && c <= '9')
        {
            scan_number_flag = 1;
            v = c - '0';
        }
        else if ( c >= 'A' && c <= 'F')
        {
            scan_number_flag = 1;
            v = c - 'A' + 10;
        }
        else if (c >= 'a' && c <= 'f')
        {
            scan_number_flag = 1;
            v = c - 'a' + 10;
        }
        else
        {
            invalid_char_flag = 1;
            break;
        }

        if (v >= base)
        {
            invalid_char_flag = 1;
            break;
        }
        result = result * base + v;
    }
    if (invalid_char_flag && endptr != NULL)
    {
        *endptr = nptr+i;
    }
    if (minus_flag < 0)
    {
        result = -result;
    }
    return result;
}
double strtod(const char *nptr, char **endptr)
{
    int i = 0;
    double last_ret = 0.0;
    double result1 = 0.0;
    double result2 = 0.0;
    int minus_flag = 1;
    int scan_number_flag = 0;
    int invalid_char_flag = 0;
    int fraction_flag = 0;
    int div = 10;
    if (nptr == NULL )
    {
        return 0.0;
    }
    for (i = 0; *(nptr+i) != '\0'; ++i)
    {
        char c = *(nptr+i);
        int v = 0;
        if (c == '+')
        {
            if (scan_number_flag)
            {
                invalid_char_flag = 1;
                break;
            }
            continue;
        }
        else if ( c == '-')
        {
            if (scan_number_flag)
            {
                invalid_char_flag = 1;
                break;
            }
            minus_flag = -minus_flag;
            continue;
        }
        else if  ( c >= '0' && c <= '9')
        {
            scan_number_flag = 1;
            v = c - '0';
        }
        else if ( c == '.')
        {
            scan_number_flag = 1;
            fraction_flag = 1;
            continue;
        }
        else
        {
            invalid_char_flag = 1;
            break;
        }

        if (fraction_flag == 0)
        {
            result1 = result1 * 10 + v;
        }
        else
        {
            result2 += (float)v / div;
            div = div * 10;
        }
    }
    if (invalid_char_flag && endptr != NULL)
    {
        *endptr = nptr+i;
    }
    last_ret = result1 + result2;
    if (minus_flag < 0)
    {
        last_ret = -last_ret;
    }
    //printf("%d.%d\n", (int)last_ret, (int)((last_ret-(int)last_ret) * 100000));
    return last_ret;
}
#define is_digit(c)	((c) >= '0' && (c) <= '9')

static int skip_atoi(const char **s)
{
	int i=0;

	while (is_digit(**s))
		i = i*10 + *((*s)++) - '0';
	return i;
}

#define ZEROPAD	1		/* pad with zero */
#define SIGN	2		/* unsigned/signed long */
#define PLUS	4		/* show plus */
#define SPACE	8		/* space if plus */
#define LEFT	16		/* left justified */
#define SPECIAL	32		/* 0x */
#define SMALL	64		/* use 'abcdef' instead of 'ABCDEF' */

#define do_div(n,base) ({ \
int __res; \
__asm__("divl %4":"=a" (n),"=d" (__res):"0" (n),"1" (0),"r" (base)); \
__res; })

static char * number(char * str, int num, int base, int size, int precision
	,int type)
{
	char c,sign,tmp[36];
	const char *digits="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	int i;

	if (type&SMALL) digits="0123456789abcdefghijklmnopqrstuvwxyz";
	if (type&LEFT) type &= ~ZEROPAD;
	if (base<2 || base>36)
		return 0;
	c = (type & ZEROPAD) ? '0' : ' ' ;
	if (type&SIGN && num<0) {
		sign='-';
		num = -num;
	} else
		sign=(type&PLUS) ? '+' : ((type&SPACE) ? ' ' : 0);
	if (sign) size--;
	if (type&SPECIAL)
		if (base==16) size -= 2;
		else if (base==8) size--;
	i=0;
	if (num==0)
		tmp[i++]='0';
	else while (num!=0)
		tmp[i++]=digits[do_div(num,base)];
	if (i>precision) precision=i;
	size -= precision;
	if (!(type&(ZEROPAD+LEFT)))
		while(size-->0)
			*str++ = ' ';
	if (sign)
		*str++ = sign;
	if (type&SPECIAL)
		if (base==8)
			*str++ = '0';
		else if (base==16) {
			*str++ = '0';
			*str++ = digits[33];
		}
	if (!(type&LEFT))
		while(size-->0)
			*str++ = c;
	while(i<precision--)
		*str++ = '0';
	while(i-->0)
		*str++ = tmp[i];
	while(size-->0)
		*str++ = ' ';
	return str;
}

int vsprintf(char *buf, size_t bufsz, const char *fmt, va_list args)
{
	int len;
	int i;
	char * str;
	char *s;
	int *ip;

	int flags;		/* flags to number() */

	int field_width;	/* width of output field */
	int precision;		/* min. # of digits for integers; max
				   number of chars for from string */
	int qualifier;		/* 'h', 'l', or 'L' for integer fields */

	for (str=buf ; *fmt && ((str-buf) < bufsz); ++fmt) {
		if (*fmt != '%') {
			*str++ = *fmt;
			continue;
		}
			
		/* process flags */
		flags = 0;
		repeat:
			++fmt;		/* this also skips first '%' */
			switch (*fmt) {
				case '-': flags |= LEFT; goto repeat;
				case '+': flags |= PLUS; goto repeat;
				case ' ': flags |= SPACE; goto repeat;
				case '#': flags |= SPECIAL; goto repeat;
				case '0': flags |= ZEROPAD; goto repeat;
				}
		
		/* get field width */
		field_width = -1;
		if (is_digit(*fmt))
			field_width = skip_atoi(&fmt);
		else if (*fmt == '*') {
			/* it's the next argument */
			field_width = va_arg(args, int);
			if (field_width < 0) {
				field_width = -field_width;
				flags |= LEFT;
			}
		}

		/* get the precision */
		precision = -1;
		if (*fmt == '.') {
			++fmt;	
			if (is_digit(*fmt))
				precision = skip_atoi(&fmt);
			else if (*fmt == '*') {
				/* it's the next argument */
				precision = va_arg(args, int);
			}
			if (precision < 0)
				precision = 0;
		}

		/* get the conversion qualifier */
		qualifier = -1;
		if (*fmt == 'h' || *fmt == 'l' || *fmt == 'L') {
			qualifier = *fmt;
			++fmt;
		}

		switch (*fmt) {
		case 'c':
			if (!(flags & LEFT))
				while (--field_width > 0)
					*str++ = ' ';
			*str++ = (unsigned char) va_arg(args, int);
			while (--field_width > 0)
				*str++ = ' ';
			break;

		case 's':
			s = va_arg(args, char *);
			len = strlen(s);
			if (precision < 0)
				precision = len;
			else if (len > precision)
				len = precision;

			if (!(flags & LEFT))
				while (len < field_width--)
					*str++ = ' ';
			for (i = 0; i < len; ++i)
				*str++ = *s++;
			while (len < field_width--)
				*str++ = ' ';
			break;

		case 'o':
			str = number(str, va_arg(args, unsigned long), 8,
				field_width, precision, flags);
			break;

		case 'p':
			if (field_width == -1) {
				field_width = 8;
				flags |= ZEROPAD;
			}
			str = number(str,
				(unsigned long) va_arg(args, void *), 16,
				field_width, precision, flags);
			break;

		case 'x':
			flags |= SMALL;
		case 'X':
			str = number(str, va_arg(args, unsigned long), 16,
				field_width, precision, flags);
			break;

		case 'd':
		case 'i':
			flags |= SIGN;
		case 'u':
			str = number(str, va_arg(args, unsigned long), 10,
				field_width, precision, flags);
			break;

		case 'n':
			ip = va_arg(args, int *);
			*ip = (str - buf);
			break;

		default:
			if (*fmt != '%')
				*str++ = '%';
			if (*fmt)
				*str++ = *fmt;
			else
				--fmt;
			break;
		}
	}
	*str = '\0';
	return str-buf;
}
