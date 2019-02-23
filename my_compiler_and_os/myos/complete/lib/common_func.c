#include "global.h"
#include "struct.h"

void init_descriptor(TDescriptor* pdes, uint32_t base, uint32_t limit, uint16_t attr)
{
	memset(pdes, 0, sizeof(TDescriptor) );

	pdes->dr_lower_limit = limit & 0xffff;
	pdes->dr_lower_base1 = base & 0xffff;
	pdes->dr_lower_base2 = (base >> 16) & 0xff;
	pdes->dr_attributes = ((limit >> 8) & 0xf00) | (attr & 0xf0ff);
	pdes->dr_higher_base = (base  >> 24) & 0xff;
}
int user_addr_to_virtual_addr(TProcess * proc, uint32_t offset, uint32_t * paddr)
{
	TDescriptor* pdes = NULL;
	uint32_t base;

	pdes = &(proc->ldts[ (proc->regs.ds) >> 3 ]);

	base = ((uint32_t)(pdes->dr_higher_base) << 24) 
	 	 + ((uint32_t)(pdes->dr_lower_base2) << 16)
	 	 + (pdes->dr_lower_base1) ;

	*paddr = base + offset;	
	return 0;
}
void init_gate(TGate* pgate, Selector s, uint32_t offset, uint8_t reserve, uint8_t attr)
{
	memset(pgate, 0, sizeof(TGate) );

#if 0
dw  (%2 & 0FFFFh)                       ; 偏移 1                (2 字节)
dw  %1                                  ; 选择子                (2 字节)
dw  (%3 & 1Fh) | ((%4 << 8) & 0FF00h)   ; 属性                  (2 字节)
dw  ((%2 >> 16) & 0FFFFh)               ; 偏移 2                (2 字节)

0x21    0xce    0x08    0x00  0x00    0xff 0x00 0x00
#endif

	pgate->gt_offset_low = offset & 0xffff;		
	pgate->gt_selector = s;
	pgate->gt_attr = (reserve & 0x1f) | ((attr << 8) & 0xff00);
	pgate->gt_offset_high = (offset >> 16) & 0xffff;
	
}
size_t strlen(const char *s)
{
	size_t ret = 0;
	if (s == NULL)
	{
		return -1;
	}
	while (s[ret] != '\0')
	{
		++ret;
	}
	return ret;
}

void memset(void * p, uint8_t v, uint32_t sz)
{
	while (sz > 0)
	{
		*((uint8_t*)p+sz-1) = v;
		--sz;
	}
}
void memcpy(void * dst, const void * src, uint32_t sz)
{
	while (sz > 0)
	{
		*((uint8_t*)dst+sz-1) = *((uint8_t*)src+sz-1);
		--sz;
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
void print_chr(char c)
{
	char * p = (char *)GS_START;
	if (c == '\n')
	{
		g_scr_line = (g_scr_line+1)%SCR_HEIGHT;
		g_scr_colume  = 0;
		return;
	}
	*(p + (g_scr_line * SCR_WIDTH + g_scr_colume) * 2) = c;
	*(p + (g_scr_line * SCR_WIDTH + g_scr_colume) * 2 + 1) = 0x0c;
	g_scr_colume = (g_scr_colume+1) % SCR_WIDTH;
}

void print_str(const char * str)
{
	int i ;
	for ( i = 0; str[i] != '\0' ; ++i)
	{
		print_chr(str[i]);
	}
}
void print_hex(uint32_t v)
{
	char buf[10];
	int_to_hex(v, buf, sizeof(buf));
	print_str(buf);
}
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

uint32_t select2index(Selector sel)
{
	return (sel >> 3);
}

static char buf[1024];
int printk(const char * fmt, ...)
{
	int len;
	va_list va;
	va_start(va, fmt);
	len = vsprintf(buf, sizeof(buf), fmt, va);
	if (len < sizeof(buf))
	{
		buf[len] = '\0';
	}
	va_end(va);
	print_str(buf);
	return len;
}
void panic(const char * fmt, ...)
{
	int len;
	va_list va;
	va_start(va, fmt);
	len = vsprintf(buf, sizeof(buf), fmt, va);
	if (len < sizeof(buf))
	{
		buf[len] = '\0';
	}
	va_end(va);
	print_str(buf);

	while (1) {};
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
