;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; �����ڴ��̵�������������С������512�ֽ�
;; ������ʵģʽ��,����ext.bin��boot.bin
;; ����0x7c00��51200�����ַ��,Ȼ����ת��0x7c00
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

%include "boot/bootsec.inc"



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

	; 1-4 ����

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

	
	;  5-17����
	mov ax, (50*1024)/16
	mov es, ax
	mov bx, 0
	mov ah, 02h
	mov al, 13 ;������Ŀ
	mov ch, 0
	mov cl, 6 ;��6������ʼ
	mov dh, 0
	mov dl, 0
	int 13h

	jnc  .conti2
	jmp .fail
.conti2:


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;��֪��Ϊ�Σ���һ�������Ĵŵ�����Ҫ�ȶ�15����Ȼ���3����������
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	sub  sp, 12
	mov bp, sp
	%define head [bp]
	%define cyl [bp+1]
	%define sect [bp+2]

	%define buf [bp+4]
	%define abs_sect [bp+6] 

	mov	word buf,  (50*1024+512*13)/16
	mov word  abs_sect,   18

.next:
	mov ax, abs_sect
	mov cl, 18
	div	cl	;AL �� Quotient, AH �� Remainder.

	;����= Q >> 1
	xor bx, bx
	mov bl, al
	shr bl, 1
	mov cyl, bl

	; ��ͷ= Q&1
	xor bx, bx
	mov bl, al
	and bl, 1
	mov head, bl

	;��ʼ���� = R+1
	xor bx, bx
	mov bl, ah
	inc bl
	mov sect, bl

	mov ax, buf
	mov es, ax
	mov bx, 0

	mov ah, 02h
	mov al, 15 ;������Ŀ
	mov ch, cyl ;�ŵ�/פ��
	mov cl, sect ;��ʼ����
	mov dh, head ;��ͷ
	mov dl, 0  ;����A
	int 13h
	jnc  .ok1
	jmp .fail

.ok1:
	add word buf, 480; 512*15/16
	add word  abs_sect, 15
	
	mov ax, abs_sect
	mov cl, 18
	div	cl	;AL �� Quotient, AH �� Remainder.

	;����= Q >> 1
	xor bx, bx
	mov bl, al
	shr bl, 1
	mov cyl, bl

	; ��ͷ= Q&1
	xor bx, bx
	mov bl, al
	and bl, 1
	mov head, bl

	;��ʼ����
	xor bx, bx
	mov bl, ah
	inc bl
	mov sect, bl

	mov ax, buf
	mov es, ax
	mov bx, 0

	mov ah, 02h
	mov al, 3 ;������Ŀ
	mov ch, cyl ;�ŵ�
	mov cl, sect ;��ʼ����
	mov dh, head ;��ͷ
	mov dl, 0
	int 13h
	jnc  .ok2
	jmp .fail

.ok2:
	add word buf, 96; 512*3/16
	add word  abs_sect, 3

	mov ax, abs_sect
	cmp ax, 90  ;һֱ������������Ϊ90�ĵط�
	jb	.next
	

	;�ر����
    mov dx,3f2h
	mov al,0
   	out dx, al


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


;INT 13H��AH=02H ������˵���� 
;���ô˹��ܽ��Ӵ����ϰ�һ���������������ݶ�������������Ϊ����һ�� 
;�ͼ����ܣ���һ�������ж�ȡ��ȫ������������ͬһ���ŵ��ϣ���ͷ�źʹŵ��� 
;��ͬ����BIOS�����Զ��ش�һ���ŵ�ĩβ�л�����һ���ŵ���ʼ������û����� 
;�ѿ�����ŵ��Ķ�������Ϊ���������ŵ��������� 
;��ڲ����� 
;AH=02H ָ�����ö��������ܡ� 
;AL ��Ҫ����������Ŀ��������ʹ�ö��ŵ�ĩ���������ֵ��Ҳ������ 
;ʹ�üĴ���Ϊ0�� 
;DL ��Ҫ���ж��������������š� 
;DH �������̵Ĵ�ͷ�š� 
;CH �ŵ��ŵĵ�8λ���� 
;CL ��5λ����������ʼ�����ţ�λ7-6��ʾ�ŵ��ŵĸ�2λ�� 
;ES:BX �������ݵĻ�������ַ�� 
;���ز����� 
;���CF=1��AX�д�ų���״̬���������������ES:BX�����������С�

;	һ��3.5��1.44M���̽ṹ 
;	
;	1�� �ṹ��2�桢80��/�桢18����/����512�ֽ�/���� 
;	         ��������=2�� X  80��/�� X  18����/��  =  2880���� 
;	         �洢����= 512�ֽ�/����X  2880���� =  1440 KB 
;	
;	2�����������ţ� 
;	         2  �棺 ���0----1�� 
;	         80���� ���0----79 
;	         18���������1----18 
;	
;	3����������ţ���2880����������������ŷ�ΧΪ 0----2879 
;	���˳�� 
;	���������               ��������� 
;	0�棬0����1����             0 
;	0�棬0����2����             1 
;	0�棬0����3����             2 
;	����������������. 
;	0�棬0����18����           17 
;	1�棬0����1����            18 
;	����������.. 
;	1�棬0����18����           35 
;	0�棬1����1����            36 
;	0�棬1����18����           53 
;	1�棬1����1����            54 
;	������ 
;	1�棬79����18����          2879 
;	
;	
