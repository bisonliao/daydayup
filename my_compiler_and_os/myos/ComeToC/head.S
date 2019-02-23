/*************************************
*至此,完成了引导,开始了内核的执行
*************************************/
.text
.globl startup_32
startup_32:
	pushl %ebp
	movl %esp, %ebp

	call func_C /*调用c代码*/

	leave
	ret

/*搞几个函数演示被c代码调用*/
/*开始新的一行的函数*/
.globl func_NewLine
func_NewLine: 
	pushl %ebp
	movl %esp,%ebp

	movl $0,Colume
	movl Line,%eax
	incl %eax
	movl %eax,Line


	movl %ebp,%esp
	popl %ebp
	ret

/*定义一个函数，用于显示一个字符*/
/*要显示的字符位于 ebp+8*/
.globl func_DispChr
func_DispChr: 
	pushl %ebp
	movl %esp,%ebp

	/*显示的位置  (Line * 80 + Colume) * 2  -> edi*/
	movl Line,%eax
	movl $80,%ebx
	mull    %ebx
	movl Colume,%ebx
	addl %ebx,%eax
	movl $2,%ebx
	mull %ebx
	movl %eax,%edi

	movl 8(%ebp),%eax   /*显示的内容从栈里获取，al*/
	movb $0xc,%ah                    /*显示的颜色*/
	movw %ax, 0x0B8000(%edi)
	/*movw %ax,%gs:%edi          #显示*/

	/*显示位置前移*/
	movl Colume,%ebx
	incl %ebx
	movl %ebx,Colume


	movl %ebp,%esp
	popl %ebp

	ret
.globl func_DispHex
func_DispHex: 
	pushl %ebp
	movl %esp,%ebp

	subl $32,%esp

	/*自动变量*/
#define    dwToShow 8(%ebp)
#define    iIndex 	-8(%ebp)

	movl $8, iIndex
.loop2: 
	movl dwToShow, %ebx
	movl iIndex, %eax
	decl %eax
	movb $4,%cl
	mulb %cl
	movb %al,%cl
	shrl %cl,%ebx
	andl $0xf,%ebx
	cmpl $9,%ebx
	ja  .above9
	addl $'0', %ebx
	jmp .print
.above9: 
	subl $10,%ebx
	addl $'A', %ebx
.print: 
	pushl %ebx
	call func_DispChr
	addl $4,%esp

	movl iIndex, %eax
	decl %eax
	movl %eax, iIndex
	cmpl $0,%eax
	jne .loop2

	movl %ebp,%esp
	popl %ebp

	ret



/* 32位数据段为程序提供字符串*/
.data
.align 32
LABEL_SEG_DATA32: 
/*屏幕显示的位置*/
	Line:   .long 1
	Colume:  .long 2
