

#if 1
/**
 * int _hd_read(uint16_t cyl, uint8_t head, uint8_t sector, unsigned char * buf)
.globl _hd_read
 */
.type _hd_read, @function
_hd_read:
	pushl %ebp
	movl %esp, %ebp

	#define cyl 8(%ebp)
	#define head 12(%ebp)
	#define sector 16(%ebp)
	#define buf 20(%ebp)

	movw	$0x1f6, %dx	#	Ҫ����Ĵ��̺ż���ͷ��
	movl	head, %eax
	andl	$0x0f, %eax
	orl		$0xa0, %eax	# bit 7 = 1��5��6��7Ϊ�̶�ֵ��
						# bit 6 = 0
						# bit 5 = 1
						# bit 4 = 0  drive 0 select��ѡ��������1��
						# = 1  drive 1 select��ѡ��������2��
						# bit 3-0    head select bits����ͷ�ţ�
	out	%al, %dx
	
	movw	$0x1f2, %dx	#Ҫ�������������
	movb	$1, %al	#��һ������
	out	%al, %dx
	
	movw	$0x1f3, %dx	#Ҫ����������
	movl	sector, %eax	#������
	out	%al, %dx
	
	movw	$0x1f4, %dx	#Ҫ��������ĵ�8λ
	movl	cyl, %eax
	out	%al, %dx
	
	movw	$0x1f5, %dx	#	�����2λ
	movl	cyl, %eax
	shrl	$8, %eax
	andl	$0x3, %eax
	out	%al, %dx
	
	movw	$0x1f7, %dx	#����˿�
	movb	$0x20, %al	#	���Զ�ȡ����
	out	%al, %dx

	movl	$100000, %ecx #�ȴ�����
_hd_read_still_going:
	xorl	%eax, %eax
	in		%dx, %al
	decl	%ecx
	cmpl	$0, %ecx
	je		_hd_read_ret_fail
	andb	$8, %al
	cmpb	$0, %al
	je		_hd_read_still_going	#�����������û��׼���õĻ�����ת��ֱ��׼���ò�����ִ�С�
	jmp		_hd_read_read_data_begin

_hd_read_ret_fail:
	movl	%ebp, %esp
	popl	%ebp
	ret

_hd_read_read_data_begin:	
	movl	$256, %ecx	#����ѭ������
	movl	buf, %ebx
_hd_read_read_data:
	xorl	%eax, %eax
	movw	$0x1f0, %dx
	inw		%dx, %ax
	movw	%ax, (%ebx)
	addl	$2, %ebx
	decl	%ecx
	cmpl	$0, %ecx
	jne		_hd_read_read_data

	movl	$0, %eax
	movl	%ebp, %esp
	popl	%ebp
	ret
/**
 * int _hd_write(uint16_t cyl, uint8_t head, uint8_t sector, const unsigned char * buf);
.globl _hd_write
 */
.type _hd_write, @function
_hd_write:
	pushl %ebp
	movl %esp, %ebp

	#define cyl 8(%ebp)
	#define head 12(%ebp)
	#define sector 16(%ebp)
	#define buf 20(%ebp)
	
	movw	$0x1f6, %dx	#	Ҫд��Ĵ��̺ż���ͷ��
	movl	head, %eax
	andl	$0x0f, %eax
	orl		$0xa0, %eax	# bit 7 = 1��5��6��7Ϊ�̶�ֵ��
						# bit 6 = 0
						# bit 5 = 1
						# bit 4 = 0  drive 0 select��ѡ��������1��
						# = 1  drive 1 select��ѡ��������2��
						# bit 3-0    head select bits����ͷ�ţ�
	out	%al, %dx
	
	movw	$0x1f2, %dx	#Ҫд�����������
	movb	$1, %al	#дһ������
	out	%al, %dx
	
	movw	$0x1f3, %dx	#Ҫд��������
	movl	sector, %eax	#������
	out	%al, %dx
	
	movw	$0x1f4, %dx	#Ҫд������ĵ�8λ
	movl	cyl, %eax
	out	%al, %dx
	
	movw	$0x1f5, %dx	#	�����2λ
	movl	cyl, %eax
	shrl	$8, %eax
	andl	$0x3, %eax
	out	%al, %dx
	
	movw	$0x1f7, %dx	#����˿�
	movb	$0x30, %al	#	����дȡ����
	out	%al, %dx
	movl	$100000, %ecx #�ȴ�����
_hd_write_still_going:
	xorl	%eax, %eax
	in		%dx, %al
	decl	%ecx
	cmpl	$0, %ecx
	je		_hd_write_ret_fail
	andb	$8, %al	#���������Ƿ�׼����
	cmpb	$0, %al
	je		_hd_write_still_going	#�����������û��׼���õĻ�����ת��ֱ��׼���ò�����ִ�С�
	jmp		_hd_write_write_data_begin

_hd_write_ret_fail:
	movl  $-1, %eax
	movl	%ebp, %esp
	popl	%ebp
	ret

_hd_write_write_data_begin:	
	movl	$256, %ecx	#����ѭ������
	movl	buf, %ebx
_hd_write_write_data:
	xorl	%eax, %eax
	movw	(%ebx), %ax
	movw	$0x1f0, %dx
	outw	%ax, %dx
	addl	$2,	%ebx
	decl	%ecx
	cmpl	$0, %ecx
	jne		_hd_write_write_data

	movl	$0, %eax
	movl	%ebp, %esp
	popl	%ebp
	ret


#endif