;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 保存在磁盘的启动扇区，大小不超过512字节
;; 运行在实模式下,负责将ext.bin和boot.bin
;; 读入0x7c00和51200物理地址处,然后跳转到0x7c00
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

    ;把自己从7c00h(31K)处挪到9000h(36K)处
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

    jmp 900h:LLL  ; 继续从挪动后的代码里运行
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


	;从磁盘读取实际待运行的代码

	mov ax, BaseOfExt
	mov es, ax
	mov bx, OffsetOfExt
	mov ah, 02h
	mov al, 4 ;扇区数目
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
	mov al, 10 ;扇区数目
	mov ch, 0
	mov cl, 6 ;从6扇区开始
	mov dh, 0
	mov dl, 0
	int 13h

	jnc  .conti2
	jmp .fail
.conti2:


	;开始运行ext.bin
	jmp dword 0:BaseOfExt*16+OffsetOfExt ;  jmp dword BaseOfExt:OffsetOfExt 这样会导致cs不为0，后面不好弄
.fail:
	mov ax, 0B800h
	mov gs, ax
	mov al, 'E'
	mov ah, 0ch
	mov edi, (80*10+2)*2
	mov [gs:edi], ax
	jmp $



