%include "pm.inc"

org 07c00h
	jmp LABEL_BEGIN

[SECTION .gdt]
; GDT
LABEL_GDT:			Descriptor 0, 0, 0
LABEL_DESC_CODE32:	Descriptor 0, SegCode32Len - 1, DA_C + DA_32
LABEL_DESC_C2:		Descriptor 0, SegC2Len - 1, DA_C + DA_32
LABEL_DESC_VIDEO:	Descriptor 0B8000h, 0ffffh, DA_DRW
LABEL_DESC_STACK:	Descriptor 20000000, 0ffffh, DA_32 + DA_DRW ; 32λģʽ�µĶ�ջ	
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


	; ��ʼ������ε�������
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

	; ��ʼ�����ݶε�������
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



end:
	jmp $

; main ������
func_main:
	push ebp
	mov  ebp, esp



	sub esp, 100

	;�Զ�����
	 ;[ss: ebp-4]   ��ʾ�����ݵ�ƫ�Ƶ�ַ
	

	;----------------------------------------
	;��ʾ data���� TestMsg������
	;���ڱ������ļ����ܴ���512�ֽڣ�ֻ����ע�͵��ⲿ��
	;----------------------------------------

	mov DWORD [ss: ebp-4], OffsetTestMsg  ;���ݵĵ�ַ�ŵ��Զ�������
	mov ecx, LengthTestMsg ;ѭ����ʾ�Ĵ���
._loop1:
	;��ʾ������ѹջ
	mov esi, [ss: ebp-4]
	mov eax, [ds:esi]
	push eax
	;���ú���
	call SelectorC2:OffsetFuncDispChr
	add  esp, 4 		;��������

	;����ָ��ǰ��
	mov eax, [ss: ebp-4]
	inc eax
	mov [ss: ebp-4], eax

	loop ._loop1


	mov  esp, ebp
	pop  ebp
	ret


SegCode32Len equ $-LABEL_SEG_CODE32

;----------------------------------------
;��һ��code��
;----------------------------------------

[SECTION .c2]
[BITS 32]
LABEL_SEG_C2:
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

	mov  eax, [ss:ebp+12] ;��ʾ�����ݴ�ջ���ȡ��al
	mov ah, 0ch			 ;��ʾ����ɫ
	mov [gs:edi], ax	 ;��ʾ

	;��ʾλ��ǰ��
	mov ebx,  [ds:OffsetColume]
	inc ebx,
	mov [ds:OffsetColume], ebx


	mov  esp, ebp
	pop  ebp

	retf ;ע�������и�f



OffsetFuncDispChr	equ func_DispChr-LABEL_SEG_C2

;��ʼ�µ�һ�еĺ���
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

; 32λ���ݶ�Ϊ�����ṩ�ַ���
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

;��Ļ��ʾ��λ��
	Line:	dd	10
OffsetLine  equ Line-LABEL_SEG_DATA32
	Colume:	 dd 0
OffsetColume equ Colume-LABEL_SEG_DATA32


SEG_DATA32_LEN	equ $ - LABEL_SEG_DATA32

