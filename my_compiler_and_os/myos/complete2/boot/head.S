/*************************************
*����,���������,��ʼ���ں˵�ִ��
*************************************/
#include "const_def.h"
.text
.globl startup_32
/**
 * ��������ں˵����,makefile
 * ��ָ����,���ұ�װ�ص� 0xc800(51200)���λ��
 */
.org 0 /*��һ��Ϊʲô��Ҫ�أ� 
        * ��ʾ����text�ڵ���ʼλ�ã����������
        * �����ŵ�text�ڵĵ�һ���ֽڴ�����֤һ
        * ����0xc800�������е�startup_32*/
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
     *
     * ������GDTת�����߼���ַ�����Ե�ַ��esp,eip����ֵ�Ͼ͵������Ե�ַ������û�жμĴ���һ����
	 */
	/*����ǰ�ĸ�descriptor */
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



	sgdt	g_gdtr48  /*��ԭ���Ļ������޸�,�ȵõ�ԭ����ֵ*/
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
	andl	$0xfffffffc, %eax  /*��4����*/
	movl	%eax, %esp	/*��ջָ��, �ںӶ�ջ��СԼ���� 30K*/


	lcall $8, $c_start	/*һ������ת���޸�csѡ����*/
	jmp .




.data
.align 32
LABEL_SEG_DATA32: 
	.long 0
	msg:.asciz ">>>>%u\n"