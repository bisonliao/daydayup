/*************************************
*����,���������,��ʼ���ں˵�ִ��
*************************************/
.text
.globl startup_32
startup_32:
	pushl %ebp
	movl %esp, %ebp

	call func_C /*����c����*/

	leave
	ret

/*�㼸��������ʾ��c�������*/
/*��ʼ�µ�һ�еĺ���*/
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

/*����һ��������������ʾһ���ַ�*/
/*Ҫ��ʾ���ַ�λ�� ebp+8*/
.globl func_DispChr
func_DispChr: 
	pushl %ebp
	movl %esp,%ebp

	/*��ʾ��λ��  (Line * 80 + Colume) * 2  -> edi*/
	movl Line,%eax
	movl $80,%ebx
	mull    %ebx
	movl Colume,%ebx
	addl %ebx,%eax
	movl $2,%ebx
	mull %ebx
	movl %eax,%edi

	movl 8(%ebp),%eax   /*��ʾ�����ݴ�ջ���ȡ��al*/
	movb $0xc,%ah                    /*��ʾ����ɫ*/
	movw %ax, 0x0B8000(%edi)
	/*movw %ax,%gs:%edi          #��ʾ*/

	/*��ʾλ��ǰ��*/
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

	/*�Զ�����*/
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



/* 32λ���ݶ�Ϊ�����ṩ�ַ���*/
.data
.align 32
LABEL_SEG_DATA32: 
/*��Ļ��ʾ��λ��*/
	Line:   .long 1
	Colume:  .long 2