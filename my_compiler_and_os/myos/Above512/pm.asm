%include "pm.inc"


;��չ���ּ��ص������ڴ�
BaseOfExt	equ 09000h
OffsetOfExt	equ 0100h
LenOfExt		equ 600


org 07c00h
	jmp LABEL_BEGIN

[SECTION .gdt]
; GDT
LABEL_GDT:			Descriptor 0, 0, 0
LABEL_DESC_CODE32:	Descriptor 0, SegCode32Len - 1, DA_C + DA_32
LABEL_DESC_VIDEO:	Descriptor 0B8000h, 0ffffh, DA_DRW
LABEL_DESC_STACK:	Descriptor 2000000, 0ffffh, DA_32 + DA_DRW ; 32λģʽ�µĶ�ջ	
LABEL_DESC_DATA32:	Descriptor 0, SEG_DATA32_LEN, DA_32 + DA_DRW
LABEL_DESC_EXT:		Descriptor (BaseOfExt*16+OffsetOfExt), LenOfExt - 1,  DA_32+DA_C
; GDT end

GdtLen 	equ $ - LABEL_GDT
GdtPtr 	dw GdtLen - 1
		dd 0

; GDT Selector
SelectorCode32	equ LABEL_DESC_CODE32 - LABEL_GDT
SelectorVideo	equ LABEL_DESC_VIDEO - LABEL_GDT
SelectorStack	equ LABEL_DESC_STACK-LABEL_GDT
SelectorData32	equ LABEL_DESC_DATA32 - LABEL_GDT
SelectorExt		equ LABEL_DESC_EXT - LABEL_GDT
;-------------------------------------------------

[SECTION .s16]
[BITS 16]
LABEL_BEGIN:
	mov ax, cs
	mov ds, ax
	mov es, ax
	mov ss, ax
	mov sp, 0100h


	; ��ʼ������ε�������
	xor eax, eax
	mov ax, cs
	shl eax, 4  ;
	add eax, LABEL_SEG_CODE32
	mov word [LABEL_DESC_CODE32+2], ax
	shr eax, 16
	mov byte [LABEL_DESC_CODE32+4], al
	mov byte [LABEL_DESC_CODE32+7], ah


	; ��ʼ�����ݶε�������
	xor eax, eax
	mov ax, ds
	shl eax, 4  ;
	add eax, LABEL_SEG_DATA32
	mov word [LABEL_DESC_DATA32+2], ax
	shr eax, 16
	mov byte [LABEL_DESC_DATA32+4], al
	mov byte [LABEL_DESC_DATA32+7], ah

	;�Ӵ��̶�ȡ��չ����
	xor ax, ax
	xor dx, dx
	int 13h

	mov ax, BaseOfExt
	mov es, ax
	mov bx, OffsetOfExt
	mov ah, 02h
	mov al, 1
	mov ch, 0
	mov cl, 2
	mov dh, 0
	mov dl, 0
	int 13h

	;�����ʵģʽ����ת��ע���޸�ext.asm
	;jmp dword BaseOfExt:OffsetOfExt


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
	;�Դ�
	mov ax, SelectorVideo
	mov gs, ax

	;����һ��ջ,���ں�������
	mov ax, SelectorStack
	mov ss, ax
	mov esp, 0ffffh 

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

	;����ӱ���ģʽ����ת��ע���޸�ext.asm
	jmp dword SelectorExt:0



	mov  esp, ebp
	pop  ebp
	ret



;��ʼ�µ�һ�еĺ���
func_NewLine:
	push ebp
	mov  ebp, esp

	mov DWORD [ds:OffsetColume], 0
	mov eax, [ds:OffsetLine]
	inc eax
	mov [ds:OffsetLine], eax
	

	mov  esp, ebp
	pop  ebp
	ret	 
	;����һ��������������ʾһ���ַ�
	;Ҫ��ʾ���ַ�λ�� ebp+8
func_DispChr:
	push ebp
	mov  ebp, esp

	;��ʾ��λ��  (Line * 80 + Colume) * 2  -> edi
	mov eax,  [ds:OffsetLine]
	mov ebx, 80
	mul	ebx
	mov ebx,  [ds:OffsetColume]
	add eax, ebx
	mov ebx, 2
	mul ebx
	mov edi, eax

	mov  eax, [ss:ebp+8] ;��ʾ�����ݴ�ջ���ȡ��al
	mov ah, 0ch			 ;��ʾ����ɫ
	mov [gs:edi], ax	 ;��ʾ

	;��ʾλ��ǰ��
	mov ebx,  [ds:OffsetColume]
	inc ebx,
	mov [ds:OffsetColume], ebx


	mov  esp, ebp
	pop  ebp

	ret 
func_DispHex:
	push ebp
	mov  ebp, esp

	sub  esp, 32

	;�Զ�����
	%define dwToShow [ss:ebp+8]
	%define i	[ss:ebp-8]

	mov DWORD i, 8
.loop2:
    mov ebx, dwToShow
    mov eax, i
    dec eax
    mov cl, 4
    mul cl
    mov cl, al
    shr ebx, cl
	and ebx, 0fh
    cmp ebx, 9
    ja  .above9
    add ebx, '0'
	jmp .print
.above9
    sub ebx, 10
    add ebx, 'A'
.print
	push ebx
	call func_DispChr
	add esp, 4

    mov eax, i
	dec eax
	mov i, eax
	cmp eax, 0
    jne .loop2
	
	mov  esp, ebp
	pop  ebp

	ret 

SegCode32Len equ $-LABEL_SEG_CODE32


; 32λ���ݶ�Ϊ�����ṩ�ַ���
[SECTION .d32]
ALIGN	32
[BITS 32]
LABEL_SEG_DATA32:
;��Ļ��ʾ��λ��
	Line:	dd	10
OffsetLine  equ Line-LABEL_SEG_DATA32
	Colume:	 dd 0
OffsetColume equ Colume-LABEL_SEG_DATA32


SEG_DATA32_LEN	equ $ - LABEL_SEG_DATA32

