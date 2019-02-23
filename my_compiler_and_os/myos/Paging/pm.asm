%include "pm.inc"

%define PT_START (PD_START+4096*10)
%define PD_START 4096000

org 07c00h
	jmp LABEL_BEGIN

[SECTION .gdt]
; GDT
LABEL_GDT:			Descriptor 0, 0, 0
LABEL_DESC_CODE32:	Descriptor 0, SegCode32Len - 1, DA_C + DA_32
LABEL_DESC_VIDEO:	Descriptor 0B8000h, 0ffffh, DA_DRW
LABEL_DESC_STACK:	Descriptor 2000000, 0ffffh, DA_32 + DA_DRW ; 32位模式下的堆栈	
LABEL_DESC_DATA32:	Descriptor 0, SEG_DATA32_LEN, DA_32 + DA_DRW
LABEL_DESC_PD:		Descriptor PD_START, 4096+4,  DA_32 + DA_DRW	;页目录
LABEL_DESC_PT:		Descriptor PT_START, 5000000,  DA_32 + DA_DRW	;页表
; GDT end

GdtLen 	equ $ - LABEL_GDT
GdtPtr 	dw GdtLen - 1
		dd 0

; GDT Selector
SelectorCode32	equ LABEL_DESC_CODE32 - LABEL_GDT
SelectorVideo	equ LABEL_DESC_VIDEO - LABEL_GDT
SelectorStack	equ LABEL_DESC_STACK-LABEL_GDT
SelectorData32	equ LABEL_DESC_DATA32 - LABEL_GDT
SelectorPD		equ LABEL_DESC_PD - LABEL_GDT
SelectorPT		equ LABEL_DESC_PT - LABEL_GDT
;-------------------------------------------------

[SECTION .s16]
[BITS 16]
LABEL_BEGIN:
	mov ax, cs
	mov ds, ax
	mov es, ax
	mov ss, ax
	mov sp, 0100h


	; 初始化代码段的描述字
	xor eax, eax
	mov ax, cs
	shl eax, 4  ;
	add eax, LABEL_SEG_CODE32
	mov word [LABEL_DESC_CODE32+2], ax
	shr eax, 16
	mov byte [LABEL_DESC_CODE32+4], al
	mov byte [LABEL_DESC_CODE32+7], ah


	; 初始化数据段的描述字
	xor eax, eax
	mov ax, ds
	shl eax, 4  ;
	add eax, LABEL_SEG_DATA32
	mov word [LABEL_DESC_DATA32+2], ax
	shr eax, 16
	mov byte [LABEL_DESC_DATA32+4], al
	mov byte [LABEL_DESC_DATA32+7], ah


	xor eax, eax
	mov ax, ds
	shl eax, 4
	add eax, LABEL_GDT
	mov dword [GdtPtr + 2], eax


	lgdt	[GdtPtr]



	cli


	in al, 92h
	or al, 00000010b
	out 92h, al


	mov eax, cr0
	or eax, 1
	mov cr0, eax

	jmp dword SelectorCode32:0

[SECTION .s32]
ALIGN	32
[BITS 32]
LABEL_SEG_CODE32:
	;显存
	mov ax, SelectorVideo
	mov gs, ax

	;开辟一格栈,用于函数调用
	mov ax, SelectorStack
	mov ss, ax
	mov esp, 0ffffh 

	;数据段
	mov ax, SelectorData32
	mov ds, ax

	;设置页表页目录
	call func_SetupPage


	;启动分页机制
	mov eax, PD_START
	mov cr3, eax
	mov eax, cr0
	or  eax, 80000000h
	mov cr0, eax


	call func_main

	jmp .end


.end:
	jmp $

; 函数
func_SetupPage:
	push ebp
	mov  ebp, esp

	mov ecx, 1024
	mov eax, PT_START + PG_P + PG_USU + PG_RWW
	mov edi, 0
	mov bx, SelectorPD
	mov es, bx
.loop
	mov [es:edi], eax
	add eax, 4096	
	add edi, 4

	loop .loop

	
%if 1
	mov ecx, 1024 * 128  ;注意这里不要超出实际物理内存大小
	mov eax, PG_P + PG_USU + PG_RWW
	mov edi, 0
	mov bx, SelectorPT
	mov es, bx
.loop2
	mov [es:edi], eax
	add eax, 4096	
	add edi, 4

	loop .loop2
%endif

	mov  esp, ebp
	pop  ebp
	ret


; main 主函数
func_main:
	push ebp
	mov  ebp, esp



	sub esp, 100

	;自动变量
	 ;[ss: ebp-4]   显示的内容的偏移地址
	

	;----------------------------------------
	;显示 data段里 TestMsg的内容
	;----------------------------------------

	mov BYTE [ds:OffsetTestMsg], '{' ;插入一个写操作

	mov DWORD [ss: ebp-4], OffsetTestMsg  ;内容的地址放到自动变量里
	mov ecx, LengthTestMsg ;循环显示的次数
._loop1:
	;显示的内容压栈
	mov esi, [ss: ebp-4]
	mov eax, [ds:esi]
	push eax
	;调用函数
	call func_DispChr
	add  esp, 4 		;弹出参数

	;内容指针前移
	mov eax, [ss: ebp-4]
	inc eax
	mov [ss: ebp-4], eax

	loop ._loop1


	mov  esp, ebp
	pop  ebp
	ret



	;定义一个函数，用于显示一个字符
	;要显示的字符位于 ebp+8
func_DispChr:
	push ebp
	mov  ebp, esp

	;显示的位置  (Line * 80 + Colume) * 2  -> edi
	mov eax,  [ds:OffsetLine]
	mov ebx, 80
	mul	ebx
	mov ebx,  [ds:OffsetColume]
	add eax, ebx
	mov ebx, 2
	mul ebx
	mov edi, eax

	mov  eax, [ss:ebp+8] ;显示的内容从栈里获取，al
	mov ah, 0ch			 ;显示的颜色
	mov [gs:edi], ax	 ;显示

	;显示位置前移
	mov ebx,  [ds:OffsetColume]
	inc ebx,
	mov [ds:OffsetColume], ebx


	mov  esp, ebp
	pop  ebp

	ret 



%if 0
;开始新的一行的函数
func_NewLine;
	push ebp
	mov  ebp, esp

	mov DWORD [ds:OffsetColume], 0
	mov eax, [ds:OffsetLine]
	inc eax
	mov [ds:OffsetLine], eax
	

	mov  esp, ebp
	pop  ebp
	ret	 
%endif

SegCode32Len equ $-LABEL_SEG_CODE32


; 32位数据段为程序提供字符串
[SECTION .d32]
ALIGN	32
[BITS 32]
LABEL_SEG_DATA32:
	PMMsg:	db " In protect mode now :)"
OffsetPMMsg 	equ	PMMsg - $$
	TestMsg:	db "[ABCDEFG"
				db "1234567890]"
LengthTestMsg	equ $-TestMsg
OffsetTestMsg	equ TestMsg - $$

;屏幕显示的位置
	Line:	dd	10
OffsetLine  equ Line-LABEL_SEG_DATA32
	Colume:	 dd 0
OffsetColume equ Colume-LABEL_SEG_DATA32


SEG_DATA32_LEN	equ $ - LABEL_SEG_DATA32

