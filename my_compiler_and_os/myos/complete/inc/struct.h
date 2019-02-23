#ifndef STRUCT_H_INCLUDED
#define STRUCT_H_INCLUDED

#include "types.h"
#include "const_def.h"



#pragma pack(1)
typedef struct
{
	uint16_t   	hd_cyl;
	uint8_t	   	hd_head;
	uint16_t	hd_reserved1;
	uint16_t	hd_wpcom;
	uint8_t		hd_reserved2;
	uint8_t		hd_ctl;
	uint8_t		hd_reserved3;
	uint8_t		hd_reserved4;
	uint8_t		hd_reserved5;
	uint16_t	hd_lzone;
	uint8_t		hd_spt; /*sects nr per track */
	uint8_t		hd_reserved6;
} THdParam;
typedef struct 
{
	uint16_t 	dr_lower_limit;
	uint16_t 	dr_lower_base1;
	uint8_t		dr_lower_base2;
	uint16_t	dr_attributes;
	uint8_t		dr_higher_base;
} TDescriptor;
typedef struct 
{
	uint16_t	gt_offset_low;	/* Offset Low */
	uint16_t	gt_selector;	/* Selector */
	uint16_t	gt_attr;		/* attribute */
	uint16_t	gt_offset_high;	/* Offset High */
}	TGate;
typedef struct 
{
	uint16_t	gr_len;	
	uint32_t	gr_base;	
} TGdtr48;
typedef struct 
{
	uint16_t	it_len;	
	uint32_t	it_base;	
} TIdtr48;


typedef struct {   
    uint32_t    gs;     
    uint32_t    fs;     
    uint32_t    es;     
    uint32_t    ds;     
    uint32_t    edi;        
    uint32_t    esi;        
    uint32_t    ebp;        
	uint32_t	temp; /* no used ? */
    uint32_t    ebx;        
    uint32_t    edx;        
    uint32_t    ecx;        
    uint32_t    eax;        
    uint32_t    retaddr;    

    uint32_t    eip;        
    uint32_t    cs;     
    uint32_t    eflags;     
    uint32_t    esp;        
    uint32_t    ss;     
}TRegContext; 	/*进程上下文保存的regs*/


typedef struct {
    TRegContext         regs;           
    Selector            ldt_sel;        
    TDescriptor         ldts[ MAX_LDT_ENT_NR ];     /* 2个就够了*/
    uint16_t            pid;            
	uint64_t			alarm; /*超时时间，当ticks大于等于该值，唤醒该进程*/
	uint8_t				status;	/*进程状态*/
	uint8_t				counter; /*时间片轮转计数器*/
	uint8_t				nice; /*优先级*/
} TProcess; 	/*进程控制块*/

typedef struct {
	uint32_t	backlink;
	uint32_t	esp0;		/* stack pointer to use during interrupt */
	uint32_t	ss0;		/*   "   segment  "  "    "        "     */
	uint32_t	esp1;
	uint32_t	ss1;
	uint32_t	esp2;
	uint32_t	ss2;
	uint32_t	cr3;
	uint32_t	eip;
	uint32_t	flags;
	uint32_t	eax;
	uint32_t	ecx;
	uint32_t	edx;
	uint32_t	ebx;
	uint32_t	esp;
	uint32_t	ebp;
	uint32_t	esi;
	uint32_t	edi;
	uint32_t	es;
	uint32_t	cs;
	uint32_t	ss;
	uint32_t	ds;
	uint32_t	fs;
	uint32_t	gs;
	uint32_t	ldt;
	uint16_t	trap;
	uint16_t	iobase;	/* I/O位图基址大于或等于TSS段界限，就表示没有I/O许可位图 */
}TSS;
#pragma pack()

#endif
