/*************************************
*至此,完成了引导,开始了内核的执行
*************************************/
#include "const_def.h"
.text
.globl startup_32
/**
 * 这里就是内核的入口,makefile
 * 里指定了,并且被装载到 0xc800(51200)这个位置
 */
.org 0 /*这一行为什么重要呢？ 
        * 表示它是text节的起始位置，避免把其他
        * 函数放到text节的第一个字节处。保证一
        * 进入0xc800就能运行到startup_32*/
startup_32:

	/**
	 * 拷贝硬盘参数到g_hd_param
	 */
#define  ParamStart 0x8c00
	xorl	%ebx, %ebx
	xorl	%edx, %edx
	movl 	$ParamStart, %edx
	movl 	$g_hd_param, %ebx
	movl	$0x10, 	%ecx
copy_base_param:
	movb	(%edx), %al
	movb	%al, (%ebx)
	incl	%edx
	incl	%ebx
	decl	%ecx
	cmpl	$0, %ecx
	jne		copy_base_param
#undef ParamStart

	/**
	 * 先重新设置GDT和相关的描述字
	 * 因为原loader设置的gdt内存访问不方便
	 * 且有可能会被覆盖，所以重新设置比较好
     *
     * 经过该GDT转换的逻辑地址和线性地址，esp,eip在数值上就等于线性地址，就像没有段寄存器一样。
	 */
	/*设置前四个descriptor */
	pushl	$0
	pushl	$0
	pushl	$0
	pushl	$g_gdt
	call	init_descriptor
	addl	$16, %esp

	/*  for g_kernel_code_selector*/
	pushl	$(DA_C|DA_32|DA_PAGE)
	pushl	$0xfffff
	pushl	$0
	pushl	$g_gdt+g_kernel_code_selector
	call	init_descriptor
	addl	$16, %esp

	/*  for g_kernel_data_selector*/
	pushl	$(DA_DRW|DA_32|DA_PAGE)
	pushl	$0xfffff
	pushl	$0
	pushl	$g_gdt+g_kernel_data_selector
	call	init_descriptor
	addl	$16, %esp

	/*  for g_kernel_gs_selector*/
	pushl	$(DA_DRW|DA_32)
	pushl	$(GS_END-GS_START)
	pushl	$(GS_START)
	pushl	$g_gdt+g_kernel_gs_selector
	call	init_descriptor
	addl	$16, %esp



	sgdt	g_gdtr48  /*在原来的基础上修改,先得到原来的值*/
	/*设置 gdt大小*/
	movw    $(8*MAX_GDT_ENT_NR-1), g_gdtr48  /*这个128等于 sizeof(TDiscriptor) * MAX_GDT_ENT_NR */
	/*设置gdt的基址*/
	movl    $g_gdt, g_gdtr48+2

	lgdt	g_gdtr48

	/* 设置selector */
	/**
	 * 注意:如果要使用c，必须将ds/es/fs/ss设置相同，否则会有莫名奇妙的问题
	 */
	movw	$16, %ax
	movw	%ax, %ds	/*数据段选择子*/
	movw	%ax, %es
	movw	%ax, %fs
	movw	%ax, %ss	/*堆栈段选择子*/

	movw	$24, %ax
	movw	%ax, %gs

	movl	$(KERNEL_ORG-1), %eax
	andl	$0xfffffffc, %eax  /*对4对齐*/
	movl	%eax, %esp	/*堆栈指针, 内河堆栈大小约等于 30K*/


	lcall $8, $c_start	/*一个长跳转，修改cs选择子*/
	jmp .




.data
.align 32
LABEL_SEG_DATA32: 
	.long 0
	msg:.asciz ">>>>%u\n"
