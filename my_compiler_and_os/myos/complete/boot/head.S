/*************************************
*����,���������,��ʼ���ں˵�ִ��
*************************************/
#include "const_def.h"
.text
.globl startup_32
startup_32:
	
	/**
	 * ����Ӳ�̲�����g_hd_param
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
	 * ����������GDT����ص�������
	 * ��Ϊԭloader���õ�gdt�ڴ���ʲ�����
	 * ���п��ܻᱻ���ǣ������������ñȽϺ�
	 */
	/*����ǰ�ĸ�descriptor */
	pushl	$0
	pushl	$0
	pushl	$0
	pushl	$g_gdt
	call	init_descriptor
	addl	$16, %esp

	pushl	$(DA_C|DA_32|DA_PAGE)
	pushl	$0xfffff
	pushl	$0
	pushl	$g_gdt+8
	call	init_descriptor
	addl	$16, %esp

	pushl	$(DA_DRW|DA_32|DA_PAGE)
	pushl	$0xfffff
	pushl	$0
	pushl	$g_gdt+16
	call	init_descriptor
	addl	$16, %esp

	pushl	$(DA_DRW|DA_32)
	pushl	$(GS_END-GS_START)
	pushl	$(GS_START)
	pushl	$g_gdt+24
	call	init_descriptor
	addl	$16, %esp

	/* ���ó��õ�selector */
	movw	$8, g_kernel_code_selector	
	movw	$16,g_kernel_data_selector
	movw	$16,g_kernel_stack_selector
	movw	$24,g_kernel_gs_selector
	movw	$32,g_idt_selector
	movw	$40,g_tss_selector


	sgdt	g_gdtr48
	/*���� gdt��С*/
	movw    $(8*MAX_GDT_ENT_NR-1), g_gdtr48  /*���128���� sizeof(TDiscriptor) * MAX_GDT_ENT_NR */
	/*����gdt�Ļ�ַ*/
	movl    $g_gdt, g_gdtr48+2

	lgdt	g_gdtr48

	/* ����selector */
	/**
	 * ע��:���Ҫʹ��c�����뽫ds/es/fs/ss������ͬ���������Ī�����������
	 */
	movw	$16, %ax
	movw	%ax, %ds	/*���ݶ�ѡ����*/
	movw	%ax, %es
	movw	%ax, %fs
	movw	%ax, %ss	/*��ջ��ѡ����*/

	movw	$24, %ax
	movw	%ax, %gs

	movl	$(KERNEL_ORG-1), %eax
	andl	$0xfffffffc, %eax
	movl	%eax, %esp	/*��ջָ��, �ںӶ�ջ��СԼ���� 30K*/

	lcall $8, $c_start	/*һ������ת���޸�csѡ����*/
	jmp .




.data
.align 32
LABEL_SEG_DATA32: 
	.long 0