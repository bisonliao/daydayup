%include "pm.inc"



org 07c00h
	jmp LABEL_BEGIN


[SECTION .s16]
[BITS 16]
LABEL_BEGIN:
	mov ax, cs
	mov ds, ax
	mov es, ax
	mov ss, ax
	mov sp, 0100h



	;�Ӵ��̶�ȡʵ�ʴ����еĴ���
	xor ax, ax
	xor dx, dx
	int 13h

	mov ax, BaseOfExt
	mov es, ax
	mov bx, OffsetOfExt
	mov ah, 02h
	mov al, 4 ;������Ŀ
	mov ch, 0
	mov cl, 2
	mov dh, 0
	mov dl, 0
	int 13h

	jmp dword BaseOfExt:OffsetOfExt



