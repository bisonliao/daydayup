;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;负责将电脑从实模式跳入保护模式,大小可以
;;突破512字节界限,但是不能超过设计上的2K
;;大小,因为只准备了4个磁盘扇区给它
;;进入保护模式后,设置分页机制,
;;然后开始执行51200处的代码,
;; 也就是boot.bin的代码
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


%include "pm.inc"


org 07c00h
	jmp LABEL_BEGIN
	;jmp $

[SECTION .gdt]
; GDT
LABEL_GDT:			Descriptor 0, 0, 0
LABEL_DESC_CODE32:	Descriptor 0, 0fffffh, DA_C + DA_32 + DA_PAGE
LABEL_DESC_VIDEO:	Descriptor 0B8000h, 0ffffh, DA_DRW
LABEL_DESC_STACK:	Descriptor 0, 0fffffh, DA_32 + DA_DRW + DA_PAGE; 32位模式下的堆栈	
LABEL_DESC_DATA32:	Descriptor 0, 0fffffh, DA_32 + DA_DRW + DA_PAGE
; GDT end

GdtLen 	equ $ - LABEL_GDT
GdtPtr 	dw GdtLen - 1
		dd 0

; GDT Selector
SelectorCode32	equ LABEL_DESC_CODE32 - LABEL_GDT
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


	jmp dword SelectorCode32:LABEL_SEG_CODE32

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
	mov esp, 0ffffh + 2000000


	;数据段
	mov ax, SelectorData32
	mov ds, ax

	;设置16M的页表
	call func_SetupPage


	call func_main

.end:
	jmp $

; 设置分页机制
func_SetupPage:
	push ebp
	mov  ebp, esp

	;设置页目录
	mov ecx, 1024
	mov eax, 4096 + PG_P + PG_USU + PG_RWW
	mov edi, 0
	mov bx, SelectorData32
	mov es, bx
.loop
	mov [es:edi], eax
	add eax, 4096	
	add edi, 4
	loop .loop

	
	;设置4个页表
	mov ecx, 1024 * 4  
	mov eax, PG_P + PG_USU + PG_RWW
	mov edi, 0
	mov bx, SelectorData32
	mov es, bx
.loop2
	mov [es:edi], eax
	add eax, 4096	
	add edi, 4

	loop .loop2

	mov eax, 0
	mov cr3, eax
	mov eax, cr0
	or  eax, 80000000h
	mov cr0, eax

	mov  esp, ebp
	pop  ebp
	ret

; main 主函数
func_main:
	push ebp
	mov  ebp, esp


	call 51200	


	mov  esp, ebp
	pop  ebp
	ret

SegCode32Len equ $-LABEL_SEG_CODE32


; 32位数据段
[SECTION .d32]
ALIGN	32
[BITS 32]
LABEL_SEG_DATA32:
times  4 db 0

SEG_DATA32_LEN	equ $ - LABEL_SEG_DATA32


