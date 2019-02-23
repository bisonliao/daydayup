;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;负责将电脑从实模式跳入保护模式,大小可以
;;突破512字节界限,但是不能超过设计上的2K
;;大小,因为只准备了4个磁盘扇区给它
;;进入保护模式后,设置分页机制,
;;然后开始执行51200处的代码,
;; 也就是boot.bin的代码
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


%include "boot/bootsec.inc"


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


; 基本参数开始的位置
BaseParamStart	equ	 8c00h

[SECTION .s16]
[BITS 16]
LABEL_BEGIN:
	mov ax, cs
	mov ds, ax
	mov es, ax
	mov ss, ax
	mov sp, 0100h
;-------------------------------------------------
;启动时BIOS会把硬盘参数表放在内存某个位置。
;对于第一个硬盘，硬盘参数表的首地址放在中断0x41处，
;即内存地址4*0x41=0x104开始的4个字节表示硬盘参数表
;的段地址(后面2字节)和偏移地址(前面2字节)
;
;第二个硬盘的参数表地址放在BIOS中断向量0x46处。
;-------------------------------------------------
	mov di, BaseParamStart
	mov ax, 0
	mov ds, ax

	mov si, [4*41h]
	mov ax, [4*41h+2]
	mov es, ax

	mov cx, 10h
.copy_hd_param:
	mov al, [es:si]
	mov [ds:di], al

	inc di
	inc si
	dec cx
	cmp cx, 0
	jne .copy_hd_param

;恢复段寄存器
	mov ax, cs
	mov es, ax
	mov ds, ax

;加载 gdtr
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

;开启保护模式
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

	call func_main

.end:
	jmp $


; main 主函数
func_main:
	push ebp
	mov  ebp, esp


	jmp 51200	


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


