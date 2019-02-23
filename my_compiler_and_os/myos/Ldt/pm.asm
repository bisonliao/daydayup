%include "pm.inc"

org 07c00h
	jmp LABEL_BEGIN

[SECTION .gdt]
; GDT
LABEL_GDT:			Descriptor 0, 0, 0
LABEL_DESC_CODE32:	Descriptor 0, SegCode32Len - 1, DA_C + DA_32
LABEL_DESC_VIDEO:	Descriptor 0B8000h, 0ffffh, DA_DRW
LABEL_DESC_TEST:	Descriptor 10000000, 0ffffh, DA_32 + DA_DRW ;访问1M以外的内存试试
LABEL_DESC_STACK:	Descriptor 20000000, 0ffffh, DA_32 + DA_DRW ; 32位模式下的堆栈	
LABEL_DESC_LDT:		Descriptor 0, LdtSegLen-1,  DA_LDT 		; ldt
; GDT end

GdtLen 	equ $ - LABEL_GDT
GdtPtr 	dw GdtLen - 1
		dd 0

; GDT Selector
SelectorCode32	equ LABEL_DESC_CODE32 - LABEL_GDT
SelectorVideo	equ LABEL_DESC_VIDEO - LABEL_GDT
SelectorTest	equ LABEL_DESC_TEST-LABEL_GDT
SelectorStack	equ LABEL_DESC_STACK-LABEL_GDT
SelectorLdt		equ LABEL_DESC_LDT - LABEL_GDT
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

	; 初始化ldt描述字
	xor eax, eax
	mov ax, ds
	shl eax, 4  ;
	add eax, LABEL_SEG_LDT
	mov word [LABEL_DESC_LDT+2], ax
	shr eax, 16
	mov byte [LABEL_DESC_LDT+4], al
	mov byte [LABEL_DESC_LDT+7], ah

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

	; test段
	mov ax, SelectorTest
	mov es, ax

	mov ax, SelectorLdt
	lldt	ax

	;数据段,必须在lldt后面  :)
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
	call func_DispChr
	add  esp, 4 		;弹出参数

	;内容指针前移
	mov eax, [ss: ebp-4]
	inc eax
	mov [ss: ebp-4], eax

	loop ._loop1

%if 0
	;----------------------------------------
	;拷贝TestStr到Test段的0偏移处
	;用到自动变量
	;[ss: ebp-8]   写入的偏移
	;[ss: ebp-12]  读取的偏移
	;[ss: ebp-16]  写的字节数
	;----------------------------------------
	mov ecx , LengthTestMsg 
	mov DWORD [ss: ebp-8], 0
	mov DWORD [ss: ebp-12], OffsetTestMsg
	mov DWORD [ss: ebp-16], 0
._loop2:
	mov esi, [ss: ebp-12]
	mov edi, [ss: ebp-8]

	mov ax, [ds:esi]
	mov [es:edi], ax

	;修改偏移
	inc esi
	inc edi
	mov [ss: ebp-12], esi
	mov [ss: ebp-8], edi

	;个数递增
	mov eax,[ss: ebp-16]
	inc eax
	mov [ss: ebp-16], eax

	loop ._loop2
	;----------------------------------------
	;换行
	;----------------------------------------
	call func_NewLine 
	;----------------------------------------
	;显示 Test里写入的内容
	;自动变量
	; [ss: ebp-20] 指示读的内容的偏移
	;----------------------------------------
	mov ecx, [ss: ebp-16]
	mov DWORD [ss: ebp-20], 0
._loop3:
	mov edi, [ss: ebp-20]
	mov eax, [es: edi]
	push eax
	call func_DispChr
	add esp, 4
	;移动指针
	mov eax,[ss: ebp-20]
	inc eax
	mov [ss: ebp-20], eax

	loop ._loop3
%endif

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

[SECTION .ldt]
ALIGN	32
LABEL_SEG_LDT:
LABEL_DESC_DATA32:	Descriptor 0, SEG_DATA32_LEN, DA_32 + DA_DRW
LdtSegLen 	equ $-LABEL_SEG_LDT

;选择子
SelectorData32	equ LABEL_DESC_DATA32-LABEL_SEG_LDT + 4


