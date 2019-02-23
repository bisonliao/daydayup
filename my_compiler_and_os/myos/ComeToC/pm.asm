;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; �����ڴ��̵�������������С������512�ֽ�
;; ������ʵģʽ��,����ext.bin��boot.bin
;; ����0x7c00��51200�����ַ��,Ȼ����ת��0x7c00
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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

    ;���Լ���7c00h(31K)��Ų��9000h(36K)��
    mov ecx, 512
    mov ax, 0
    mov es, ax
    mov ds, ax
    mov esi, 7c00h
    mov edi, 9000h
.loop:
    mov al, [es:esi]
    mov [ds:edi], al
    inc esi
    inc edi
	dec ecx
	cmp ecx, 0
	jne .loop

    jmp 900h:LLL  ; ������Ų����Ĵ���������
	;jmp  0:9000h + LLL

	LLL equ $-07c00h

	mov ax,cs
	mov ds,ax
	mov es,ax
	mov ss,ax
	mov sp, 0f00h      ; arbitrary value >>512

	xor ax, ax
	xor dx, dx
	int 13h


	;�Ӵ��̶�ȡʵ�ʴ����еĴ���

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

	jnc  .conti1
	jmp .fail
.conti1:
	


	mov ax, 50*1024/16
	mov es, ax
	mov bx, 0
	mov ah, 02h
	mov al, 10 ;������Ŀ
	mov ch, 0
	mov cl, 6 ;��6������ʼ
	mov dh, 0
	mov dl, 0
	int 13h

	jnc  .conti2
	jmp .fail
.conti2:


	;��ʼ����ext.bin
	jmp dword 0:BaseOfExt*16+OffsetOfExt ;  jmp dword BaseOfExt:OffsetOfExt �����ᵼ��cs��Ϊ0�����治��Ū
.fail:
	mov ax, 0B800h
	mov gs, ax
	mov al, 'E'
	mov ah, 0ch
	mov edi, (80*10+2)*2
	mov [gs:edi], ax
	jmp $



