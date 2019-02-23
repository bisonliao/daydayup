;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;���𽫵��Դ�ʵģʽ���뱣��ģʽ,��С����
;;ͻ��512�ֽڽ���,���ǲ��ܳ�������ϵ�2K
;;��С,��Ϊֻ׼����4��������������
;;���뱣��ģʽ��,���÷�ҳ����,
;;Ȼ��ʼִ��51200���Ĵ���,
;; Ҳ����boot.bin�Ĵ���
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
LABEL_DESC_STACK:	Descriptor 0, 0fffffh, DA_32 + DA_DRW + DA_PAGE; 32λģʽ�µĶ�ջ	
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


; ����������ʼ��λ��
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
;����ʱBIOS���Ӳ�̲���������ڴ�ĳ��λ�á�
;���ڵ�һ��Ӳ�̣�Ӳ�̲�������׵�ַ�����ж�0x41����
;���ڴ��ַ4*0x41=0x104��ʼ��4���ֽڱ�ʾӲ�̲�����
;�Ķε�ַ(����2�ֽ�)��ƫ�Ƶ�ַ(ǰ��2�ֽ�)
;
;�ڶ���Ӳ�̵Ĳ������ַ����BIOS�ж�����0x46����
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

;�ָ��μĴ���
	mov ax, cs
	mov es, ax
	mov ds, ax

;���� gdtr
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

;��������ģʽ
	mov eax, cr0
	or eax, 1
	mov cr0, eax


	jmp dword SelectorCode32:LABEL_SEG_CODE32

[SECTION .s32]
ALIGN	32
[BITS 32]
LABEL_SEG_CODE32:
	;�Դ�
	mov ax, SelectorVideo
	mov gs, ax

	;����һ��ջ,���ں�������
	mov ax, SelectorStack
	mov ss, ax
	mov esp, 0ffffh + 2000000


	;���ݶ�
	mov ax, SelectorData32
	mov ds, ax

	call func_main

.end:
	jmp $


; main ������
func_main:
	push ebp
	mov  ebp, esp


	jmp 51200	


	mov  esp, ebp
	pop  ebp
	ret

SegCode32Len equ $-LABEL_SEG_CODE32


; 32λ���ݶ�
[SECTION .d32]
ALIGN	32
[BITS 32]
LABEL_SEG_DATA32:
times  4 db 0

SEG_DATA32_LEN	equ $ - LABEL_SEG_DATA32


