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
LABEL_DESC_STACK:	Descriptor 2000000, 0ffffh, DA_32 + DA_DRW ; 32λģʽ�µĶ�ջ	
LABEL_DESC_DATA32:	Descriptor 0, SEG_DATA32_LEN, DA_32 + DA_DRW
LABEL_DESC_PD:		Descriptor PD_START, 4096+4,  DA_32 + DA_DRW	;ҳĿ¼
LABEL_DESC_PT:		Descriptor PT_START, 5000000,  DA_32 + DA_DRW	;ҳ��
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

	;����ҳ��ҳĿ¼
	call func_SetupPage


	;������ҳ����
	mov eax, PD_START
	mov cr3, eax
	mov eax, cr0
	or  eax, 80000000h
	mov cr0, eax


	call func_main

	jmp .end


.end:
	jmp $

; ����
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
	mov ecx, 1024 * 128  ;ע�����ﲻҪ����ʵ�������ڴ��С
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


; main ������
func_main:
	push ebp
	mov  ebp, esp



	sub esp, 100

	;�Զ�����
	 ;[ss: ebp-4]   ��ʾ�����ݵ�ƫ�Ƶ�ַ
	

	;----------------------------------------
	;��ʾ data���� TestMsg������
	;----------------------------------------

	mov BYTE [ds:OffsetTestMsg], '{' ;����һ��д����

	mov DWORD [ss: ebp-4], OffsetTestMsg  ;���ݵĵ�ַ�ŵ��Զ�������
	mov ecx, LengthTestMsg ;ѭ����ʾ�Ĵ���
._loop1:
	;��ʾ������ѹջ
	mov esi, [ss: ebp-4]
	mov eax, [ds:esi]
	push eax
	;���ú���
	call func_DispChr
	add  esp, 4 		;��������

	;����ָ��ǰ��
	mov eax, [ss: ebp-4]
	inc eax
	mov [ss: ebp-4], eax

	loop ._loop1


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



%if 0
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
%endif

SegCode32Len equ $-LABEL_SEG_CODE32


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

