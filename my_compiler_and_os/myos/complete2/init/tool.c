#include "global.h"
#include "struct.h"
#include "redefine.h"

void init_descriptor(TDescriptor* pdes, uint32_t base, uint32_t limit, uint16_t attr)
{
	memset(pdes, 0, sizeof(TDescriptor) );

	pdes->dr_lower_limit = limit & 0xffff;
	pdes->dr_lower_base1 = base & 0xffff;
	pdes->dr_lower_base2 = (base >> 16) & 0xff;
	pdes->dr_attributes = ((limit >> 8) & 0xf00) | (attr & 0xf0ff);
	pdes->dr_higher_base = (base  >> 24) & 0xff;
}
int paddr_to_user_space_vaddr(uint32_t pid, uint32_t offset, uint32_t *vaddr)
{
    if (pid < 1 || pid > MAX_PROC_NR)
    {
		panic("%s %d: invalid pid %u\n", __FILE__, __LINE__, pid);
    }
    uint32_t user_space_org = FIRST_PROC_ORG+ (pid-1)*PROC_SPACE; /*进程的用户态空间的起始物理地址*/
    if (offset < user_space_org || offset > (user_space_org+PROC_SPACE) )
    {
		panic("%s %d: invalid offset %u\n", __FILE__, __LINE__, offset);
    }
    *vaddr = USER_SPACE_VADDR_HEAD + (offset - user_space_org);
    return 0;
}
#if 1
/*用户态的虚拟地址转化为物理地址*/
int user_space_vaddr_to_paddr(TProcess * proc, uint32_t offset, uint32_t * paddr, uint32_t ds)
{
    if (proc->pid < 1 || proc->pid > MAX_PROC_NR)
    {
		panic("%s %d: invalid pid %u\n", __FILE__, __LINE__, proc->pid);
    }
    if (_get_cr3() != 0) 
    {
        panic("%s %d: please set cr3 to 0 first!\n", __FILE__, __LINE__);
    }
    uint32_t user_space_org = FIRST_PROC_ORG+ (proc->pid-1)*PROC_SPACE; /*进程的用户态空间的起始物理地址*/
    *paddr = user_space_org + offset - USER_SPACE_VADDR_HEAD;
    return 0;
}
#endif
#if 0
int user_addr_to_virtual_addr(TProcess * proc, uint32_t offset, uint32_t * paddr, uint32_t ds)
{
	TDescriptor* pdes = NULL;
	uint32_t base;

	/**
	 * 这里为什么不直接使用proc->tss.ds而需要额外传一个参数ds呢？
	 * 因为proc->tss.ds只是最近一次从其他进程切换到当前进程时的选择子,
	 * 也是当前进程最近一次被切换出去时候保存的选择子.
	 * 这个选择子有两种可能：进程的内核态选择子和用户态选择子，两者不相等。
     * 因为进程运行在内核态的时候也可以发生切换，比如在等待IO发生时
	 * 而当前函数明确需要用户态选择子，所以需要额外传递参数.
	 * 这个错误的纠正花了半天时间。
	 */

	/*
	if (ds != proc->tss.ds)
	{
		printk("!= %d, %d, %d\n", ds, proc->tss.ds, proc->pid);
		while (1) {};
	}
	*/
    /*todo:检查一下offset是否有效*/

	pdes = &(proc->ldts[ ds >> 3 ]);
	//pdes = &(proc->ldts[ (proc->tss.ds & 0xff) >> 3 ]);

	base = ((uint32_t)(pdes->dr_higher_base) << 24) 
	 	 + ((uint32_t)(pdes->dr_lower_base2) << 16)
	 	 + (pdes->dr_lower_base1) ;

	*paddr = base + offset;	
	return 0;
}
#endif
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

