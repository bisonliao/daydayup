

#if 1
/**
 * int _hd_read(uint16_t cyl, uint8_t head, uint8_t sector, unsigned char * buf)
 */
.globl _hd_read
.type _hd_read, @function
_hd_read:
	pushl %ebp
	movl %esp, %ebp

	#define cyl 8(%ebp)
	#define head 12(%ebp)
	#define sector 16(%ebp)
	#define buf 20(%ebp)

	movw	$0x1f6, %dx	#	要读入的磁盘号及磁头号
	movl	head, %eax
	andl	$0x0f, %eax
	orl		$0xa0, %eax	# bit 7 = 1（5，6，7为固定值）
						# bit 6 = 0
						# bit 5 = 1
						# bit 4 = 0  drive 0 select（选择驱动器1）
						# = 1  drive 1 select（选择驱动器2）
						# bit 3-0    head select bits（磁头号）
	out	%al, %dx
	
	movw	$0x1f2, %dx	#要读入的扇区数量
	movb	$1, %al	#读一个扇区
	out	%al, %dx
	
	movw	$0x1f3, %dx	#要读的扇区号
	movl	sector, %eax	#扇区号
	out	%al, %dx
	
	movw	$0x1f4, %dx	#要读的柱面的低8位
	movl	cyl, %eax
	out	%al, %dx
	
	movw	$0x1f5, %dx	#	柱面高2位
	movl	cyl, %eax
	shrl	$8, %eax
	andl	$0x3, %eax
	out	%al, %dx
	
	movw	$0x1f7, %dx	#命令端口
	movb	$0x20, %al	#	尝试读取扇区
	out	%al, %dx

	movl	$100000, %ecx #等待次数
_hd_read_still_going:
	xorl	%eax, %eax
	in		%dx, %al
	decl	%ecx
	cmpl	$0, %ecx
	je		_hd_read_ret_fail
	andb	$8, %al
	cmpb	$0, %al
	je		_hd_read_still_going	#如果扇区缓冲没有准备好的话则跳转，直到准备好才向下执行。
	jmp		_hd_read_read_data_begin

_hd_read_ret_fail:
	movl	%ebp, %esp
	popl	%ebp
	ret

_hd_read_read_data_begin:	
	movl	$256, %ecx	#设置循环次数
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
 */
.globl _hd_write
.type _hd_write, @function
_hd_write:
	pushl %ebp
	movl %esp, %ebp

	#define cyl 8(%ebp)
	#define head 12(%ebp)
	#define sector 16(%ebp)
	#define buf 20(%ebp)
	
	movw	$0x1f6, %dx	#	要写入的磁盘号及磁头号
	movl	head, %eax
	andl	$0x0f, %eax
	orl		$0xa0, %eax	# bit 7 = 1（5，6，7为固定值）
						# bit 6 = 0
						# bit 5 = 1
						# bit 4 = 0  drive 0 select（选择驱动器1）
						# = 1  drive 1 select（选择驱动器2）
						# bit 3-0    head select bits（磁头号）
	out	%al, %dx
	
	movw	$0x1f2, %dx	#要写入的扇区数量
	movb	$1, %al	#写一个扇区
	out	%al, %dx
	
	movw	$0x1f3, %dx	#要写的扇区号
	movl	sector, %eax	#扇区号
	out	%al, %dx
	
	movw	$0x1f4, %dx	#要写的柱面的低8位
	movl	cyl, %eax
	out	%al, %dx
	
	movw	$0x1f5, %dx	#	柱面高2位
	movl	cyl, %eax
	shrl	$8, %eax
	andl	$0x3, %eax
	out	%al, %dx
	
	movw	$0x1f7, %dx	#命令端口
	movb	$0x30, %al	#	尝试写取扇区
	out	%al, %dx
	movl	$100000, %ecx #等待次数
_hd_write_still_going:
	xorl	%eax, %eax
	in		%dx, %al
	decl	%ecx
	cmpl	$0, %ecx
	je		_hd_write_ret_fail
	andb	$8, %al	#扇区缓冲是否准备好
	cmpb	$0, %al
	je		_hd_write_still_going	#如果扇区缓冲没有准备好的话则跳转，直到准备好才向下执行。
	jmp		_hd_write_write_data_begin

_hd_write_ret_fail:
	movl  $-1, %eax
	movl	%ebp, %esp
	popl	%ebp
	ret

_hd_write_write_data_begin:	
	movl	$256, %ecx	#设置循环次数
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
