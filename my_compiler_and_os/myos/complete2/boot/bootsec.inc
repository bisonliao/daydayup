
%macro Descriptor 3
	dw %2 & 0ffffh
	dw %1 & 0ffffh
	db (%1 >> 16) & 0ffh
	dw ((%2 >> 8)& 0f00h) | (%3 & 0f0ffh)
	db (%1 >> 24) & 0ffh
%endmacro

DA_32		EQU	4000h	; 32 λ��
DA_DRW		EQU	92h	; ���ڵĿɶ�д���ݶ�����ֵ
DA_C		EQU	98h	; ���ڵ�ִֻ�д��������ֵ
DA_LDT		EQU	  82h	; �ֲ���������������ֵ
DA_PAGE		EQU	8000h	;�ν��޵�����Ϊ4K

PG_P		equ	01h
PG_USU		equ 04h
PG_RWW		equ 02h

;��������ص������ڴ�07c00h��ʼ��λ�ã���2K
BaseOfExt   equ 07c0h
OffsetOfExt equ 0

BaseOfBoot   equ 2112
OffsetOfBoot equ 0

