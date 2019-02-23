%include "pm.inc"

org 07c00h
	jmp LABEL_BEGIN

[SECTION .gdt]
; GDT
LABEL_GDT:			Descriptor 0, 0, 0
LABEL_DESC_CODE32:	Descriptor 0, SegCode32Len - 1, DA_C + DA_32
LABEL_DESC_C2:		Descriptor 0, SegC2Len - 1, DA_C + DA_32
LABEL_DESC_VIDEO:	Descriptor 0B8000h, 0ffffh, DA_DRW
LABEL_DESC_STACK:	Descriptor 20000000, 0ffffh, DA_32 + DA_DRW ; 32位模式下的堆栈	
LABEL_DESC_DATA32:	Descriptor 0, SEG_DATA32_LEN, DA_32 + DA_DRW
; GDT end

GdtLen 	equ $ - LABEL_GDT
GdtPtr 	dw GdtLen - 1
		dd 0

; GDT Selector
SelectorCode32	equ LABEL_DESC_CODE32 - LABEL_GDT
SelectorC2		equ LABEL_DESC_C2 - LABEL_GDT
SelectorVideo	equ LABEL_DESC_VIDEO - LABEL_GDT
SelectorStack	equ LABEL_DESC_STACK-LABEL_GDT
SelectorData32	equ LABEL_DESC_DATA32 - LABEL_GDT
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

	xor eax, eax
	mov ax, cs
	shl eax, 4  ;
	add eax, LABEL_SEG_C2
	mov word [LABEL_DESC_C2+2], ax
	shr eax, 16
	mov byte [LABEL_DESC_C2+4], al
	mov byte [LABEL_DESC_C2+7], ah

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

	call func_main



end:
	jmp $

; main 主函数
func_main:
	push ebp
	mov  ebp, esp



	sub esp, 100

	;自动变量
	 ;[ss: ebp-4]   显示的内容的偏移地址
	

	;----------------------------------------
	;显示 data段里 TestMsg的内容
	;由于编译后的文件不能大于512字节，只好先注释掉这部分
	;----------------------------------------

	mov DWORD [ss: ebp-4], OffsetTestMsg  ;内容的地址放到自动变量里
	mov ecx, LengthTestMsg ;循环显示的次数
._loop1:
	;显示的内容压栈
	mov esi, [ss: ebp-4]
	mov eax, [ds:esi]
	push eax
	;调用函数
	call SelectorC2:OffsetFuncDispChr
	add  esp, 4 		;弹出参数

	;内容指针前移
	mov eax, [ss: ebp-4]
	inc eax
	mov [ss: ebp-4], eax

	loop ._loop1


	mov  esp, ebp
	pop  ebp
	ret


SegCode32Len equ $-LABEL_SEG_CODE32

;----------------------------------------
;另一个code段
;----------------------------------------

[SECTION .c2]
[BITS 32]
LABEL_SEG_C2:
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

	mov  eax, [ss:ebp+12] ;显示的内容从栈里获取，al
	mov ah, 0ch			 ;显示的颜色
	mov [gs:edi], ax	 ;显示

	;显示位置前移
	mov ebx,  [ds:OffsetColume]
	inc ebx,
	mov [ds:OffsetColume], ebx


	mov  esp, ebp
	pop  ebp

	retf ;注意这里有个f



OffsetFuncDispChr	equ func_DispChr-LABEL_SEG_C2

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

OffsetFuncDispNewLine	equ func_NewLine-LABEL_SEG_C2


SegC2Len equ $-LABEL_SEG_C2

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

