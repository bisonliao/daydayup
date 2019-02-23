%include "pm.inc"

org 07c00h
	jmp LABEL_BEGIN

[SECTION .gdt]
; GDT
LABEL_GDT:			Descriptor 0, 0, 0
LABEL_DESC_CODE32:	Descriptor 0, SegCode32Len - 1, DA_C + DA_32
LABEL_DESC_VIDEO:	Descriptor 0B8000h, 0ffffh, DA_DRW
LABEL_DESC_TEST:	Descriptor 10000000, 0ffffh, DA_32 + DA_DRW ;����1M������ڴ�����
LABEL_DESC_STACK:	Descriptor 20000000, 0ffffh, DA_32 + DA_DRW ; 32λģʽ�µĶ�ջ	
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

	; ��ʼ��ldt������
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
	;�Դ�
	mov ax, SelectorVideo
	mov gs, ax

	;����һ��ջ,���ں�������
	mov ax, SelectorStack
	mov ss, ax
	mov esp, 0ffffh 

	; test��
	mov ax, SelectorTest
	mov es, ax

	mov ax, SelectorLdt
	lldt	ax

	;���ݶ�,������lldt����  :)
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
	call func_DispChr
	add  esp, 4 		;��������

	;����ָ��ǰ��
	mov eax, [ss: ebp-4]
	inc eax
	mov [ss: ebp-4], eax

	loop ._loop1

%if 0
	;----------------------------------------
	;����TestStr��Test�ε�0ƫ�ƴ�
	;�õ��Զ�����
	;[ss: ebp-8]   д���ƫ��
	;[ss: ebp-12]  ��ȡ��ƫ��
	;[ss: ebp-16]  д���ֽ���
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

	;�޸�ƫ��
	inc esi
	inc edi
	mov [ss: ebp-12], esi
	mov [ss: ebp-8], edi

	;��������
	mov eax,[ss: ebp-16]
	inc eax
	mov [ss: ebp-16], eax

	loop ._loop2
	;----------------------------------------
	;����
	;----------------------------------------
	call func_NewLine 
	;----------------------------------------
	;��ʾ Test��д�������
	;�Զ�����
	; [ss: ebp-20] ָʾ�������ݵ�ƫ��
	;----------------------------------------
	mov ecx, [ss: ebp-16]
	mov DWORD [ss: ebp-20], 0
._loop3:
	mov edi, [ss: ebp-20]
	mov eax, [es: edi]
	push eax
	call func_DispChr
	add esp, 4
	;�ƶ�ָ��
	mov eax,[ss: ebp-20]
	inc eax
	mov [ss: ebp-20], eax

	loop ._loop3
%endif

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

[SECTION .ldt]
ALIGN	32
LABEL_SEG_LDT:
LABEL_DESC_DATA32:	Descriptor 0, SEG_DATA32_LEN, DA_32 + DA_DRW
LdtSegLen 	equ $-LABEL_SEG_LDT

;ѡ����
SelectorData32	equ LABEL_DESC_DATA32-LABEL_SEG_LDT + 4


