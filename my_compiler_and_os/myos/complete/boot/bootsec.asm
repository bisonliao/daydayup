;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 保存在磁盘的启动扇区，大小不超过512字节
;; 运行在实模式下,负责将ext.bin和boot.bin
;; 读入0x7c00和51200物理地址处,然后跳转到0x7c00
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

	; 1-4 扇区

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

	
	;  5-17扇区
	mov ax, (50*1024)/16
	mov es, ax
	mov bx, 0
	mov ah, 02h
	mov al, 13 ;扇区数目
	mov ch, 0
	mov cl, 6 ;从6扇区开始
	mov dh, 0
	mov dl, 0
	int 13h

	jnc  .conti2
	jmp .fail
.conti2:


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;不知道为何，读一个完整的磁道必须要先读15扇区然后读3个扇区才能
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
	div	cl	;AL ← Quotient, AH ← Remainder.

	;柱面= Q >> 1
	xor bx, bx
	mov bl, al
	shr bl, 1
	mov cyl, bl

	; 磁头= Q&1
	xor bx, bx
	mov bl, al
	and bl, 1
	mov head, bl

	;起始扇区 = R+1
	xor bx, bx
	mov bl, ah
	inc bl
	mov sect, bl

	mov ax, buf
	mov es, ax
	mov bx, 0

	mov ah, 02h
	mov al, 15 ;扇区数目
	mov ch, cyl ;磁道/驻缅
	mov cl, sect ;起始扇区
	mov dh, head ;磁头
	mov dl, 0  ;磁盘A
	int 13h
	jnc  .ok1
	jmp .fail

.ok1:
	add word buf, 480; 512*15/16
	add word  abs_sect, 15
	
	mov ax, abs_sect
	mov cl, 18
	div	cl	;AL ← Quotient, AH ← Remainder.

	;柱面= Q >> 1
	xor bx, bx
	mov bl, al
	shr bl, 1
	mov cyl, bl

	; 磁头= Q&1
	xor bx, bx
	mov bl, al
	and bl, 1
	mov head, bl

	;起始扇区
	xor bx, bx
	mov bl, ah
	inc bl
	mov sect, bl

	mov ax, buf
	mov es, ax
	mov bx, 0

	mov ah, 02h
	mov al, 3 ;扇区数目
	mov ch, cyl ;磁道
	mov cl, sect ;起始扇区
	mov dh, head ;磁头
	mov dl, 0
	int 13h
	jnc  .ok2
	jmp .fail

.ok2:
	add word buf, 96; 512*3/16
	add word  abs_sect, 3

	mov ax, abs_sect
	cmp ax, 90  ;一直读到绝对扇区为90的地方
	jb	.next
	

	;关闭马达
    mov dx,3f2h
	mov al,0
   	out dx, al


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


;INT 13H，AH=02H 读扇区说明： 
;调用此功能将从磁盘上把一个或更多的扇区内容读进存贮器。因为这是一个 
;低级功能，在一个操作中读取的全部扇区必须在同一条磁道上（磁头号和磁道号 
;相同）。BIOS不能自动地从一条磁道末尾切换到另一条磁道开始，因此用户必须 
;把跨多条磁道的读操作分为若干条单磁道读操作。 
;入口参数： 
;AH=02H 指明调用读扇区功能。 
;AL 置要读的扇区数目，不允许使用读磁道末端以外的数值，也不允许 
;使该寄存器为0。 
;DL 需要进行读操作的驱动器号。 
;DH 所读磁盘的磁头号。 
;CH 磁道号的低8位数。 
;CL 低5位放入所读起始扇区号，位7-6表示磁道号的高2位。 
;ES:BX 读出数据的缓冲区地址。 
;返回参数： 
;如果CF=1，AX中存放出错状态。读出后的数据在ES:BX区域依次排列。

;	一、3.5寸1.44M软盘结构 
;	
;	1、 结构：2面、80道/面、18扇区/道、512字节/扇区 
;	         扇区总数=2面 X  80道/面 X  18扇区/道  =  2880扇区 
;	         存储容量= 512字节/扇区X  2880扇区 =  1440 KB 
;	
;	2、物理扇区号： 
;	         2  面： 编号0----1； 
;	         80道： 编号0----79 
;	         18扇区：编号1----18 
;	
;	3、相对扇区号：共2880个扇区，相对扇区号范围为 0----2879 
;	编号顺序： 
;	扇区物理号               相对扇区号 
;	0面，0道，1扇区             0 
;	0面，0道，2扇区             1 
;	0面，0道，3扇区             2 
;	……………………. 
;	0面，0道，18扇区           17 
;	1面，0道，1扇区            18 
;	…………….. 
;	1面，0道，18扇区           35 
;	0面，1道，1扇区            36 
;	0面，1道，18扇区           53 
;	1面，1道，1扇区            54 
;	……… 
;	1面，79道，18扇区          2879 
;	
;	
