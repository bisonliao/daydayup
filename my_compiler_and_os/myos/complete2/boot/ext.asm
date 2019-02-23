;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;���𽫵��Դ�ʵģʽ���뱣��ģʽ,��С����
;;ͻ��512�ֽڽ���,���ǲ��ܳ�������ϵ�2K
;;��С,��Ϊֻ׼����4��������������
;;���뱣��ģʽ��,���÷�ҳ����,
;;Ȼ��ʼִ��51200���Ĵ���,
;; Ҳ����boot.bin�Ĵ���
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


%include "boot/bootsec.inc"


org 07c00h
	jmp LABEL_BEGIN
	;jmp $

[SECTION .gdt]
; GDT
; ��Ҫ��ÿ���ε���ʼ��ַ�����ȡ�����Ȩ�޵���Ϣ
; data/stack/code����ʼ��ַ����0,���ȶ���4Gȫ���ڴ�
; ������GDTת�����߼���ַ�����Ե�ַ��esp,eip����ֵ�Ͼ͵������Ե�ַ������û�жμĴ���һ����
; ����eip/esp�ȼĴ�����ĵ�ַ�͵������Ե�ַ
; ͨ����һ����,�ƹ���x86�ķֶλ���
; baseaddr,lenlimit,property
LABEL_GDT:			Descriptor 0, 0, 0                              ;��һ����ʹ��
LABEL_DESC_CODE32:	Descriptor 0, 0fffffh, DA_C + DA_32 + DA_PAGE   ;32λ����Σ� 4G��ȫ����ַ�ռ�
LABEL_DESC_VIDEO:	Descriptor 0B8000h, 0ffffh, DA_DRW              ;�Դ�
LABEL_DESC_STACK:	Descriptor 0, 0fffffh, DA_32 + DA_DRW + DA_PAGE ;32λģʽ�µĶ�ջ 4G��ȫ����ַ�ռ�
LABEL_DESC_DATA32:	Descriptor 0, 0fffffh, DA_32 + DA_DRW + DA_PAGE ;32λ���ݶ� 4G��ȫ����ַ�ռ�
; GDT end

GdtLen 	equ $ - LABEL_GDT
; 6�ֽڵ�һ�����ˣ���lgdtָ����
GdtPtr 	dw GdtLen - 1   ;ǰ�����ֽڴ泤��-1
		dd 0            ;����4�ֽڽ����GDT�ľ��������ַ

; GDT Selector
; ѡ���ӣ����ڶ�ӦGDT�����ƫ����
SelectorCode32	equ LABEL_DESC_CODE32 - LABEL_GDT  
SelectorVideo	equ LABEL_DESC_VIDEO - LABEL_GDT
SelectorStack	equ LABEL_DESC_STACK-LABEL_GDT
SelectorData32	equ LABEL_DESC_DATA32 - LABEL_GDT
;-------------------------------------------------


; ����������ʼ��λ��
BaseParamStart	equ	 8c00h

[SECTION .s16]
[BITS 16]
LABEL_BEGIN:
	mov ax, cs
	mov ds, ax
	mov es, ax
	mov ss, ax
	mov sp, 0100h
;-------------------------------------------------
;����ʱBIOS���Ӳ�̲���������ڴ�ĳ��λ�á�
;���ڵ�һ��Ӳ�̣�Ӳ�̲�������׵�ַ�����ж�0x41����
;���ڴ��ַ4*0x41=0x104��ʼ��4���ֽڱ�ʾӲ�̲�����
;�Ķε�ַ(����2�ֽ�)��ƫ�Ƶ�ַ(ǰ��2�ֽ�)
;
;�ڶ���Ӳ�̵Ĳ������ַ����BIOS�ж�����0x46����
;-------------------------------------------------
	mov di, BaseParamStart
	mov ax, 0
	mov ds, ax

	mov si, [4*41h]   ;ƫ�Ƶ�ַ
	mov ax, [4*41h+2] ;�ε�ַ
	mov es, ax

	mov cx, 10h  ;�����ܳ���16�ֽ�?
.copy_hd_param:
	mov al, [es:si]
	mov [ds:di], al

	inc di
	inc si
	dec cx
	cmp cx, 0
	jne .copy_hd_param

;�ָ��μĴ���
	mov ax, cs
	mov es, ax
	mov ds, ax

;���� gdtr
	xor eax, eax
    ;���μĴ�������*16 + ƫ�� �͵õ�LABEL_GDT�ľ��������ַ
    ;ʵģʽ�µĵ�ַ��������������������
	mov ax, ds
	shl eax, 4
	add eax, LABEL_GDT
	mov dword [GdtPtr + 2], eax

	lgdt	[GdtPtr]

	cli     ;���ж�

    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    ;����ʵģʽ�½���20 ����ַ�ߣ�A0, A1, . . . , A19��
    ;���Ե�����Ҫ���뱣��ģʽʱ����Ҫ��A20��ַ��
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	in al, 92h
	or al, 00000010b
	out 92h, al

;��������ģʽ
	mov eax, cr0
	or eax, 1
	mov cr0, eax


	jmp dword SelectorCode32:LABEL_SEG_CODE32  ; cs�Ĵ����ڸ�ָ����ɺ������ΪSelectorCode32��

[SECTION .s32]
ALIGN	32
[BITS 32]
LABEL_SEG_CODE32:
	;�Դ�
	mov ax, SelectorVideo
	mov gs, ax

	;����һ��ջ,���ں�������
    ; GDT�����ã�0-4G��ַ�ռ�Ϊջ�ռ䣬ʵ��ʹ��0-2M���һ��
    ; ѹջʱ�Ӹߵ�ַ���͵�ַ�仯
	mov ax, SelectorStack
	mov ss, ax
	mov esp, 0ffffh + 2000000   


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


	jmp 51200	 ;�ں˾ͱ�װ�������λ��,bootsec.asm�����������


	mov  esp, ebp
	pop  ebp
	ret

SegCode32Len equ $-LABEL_SEG_CODE32


; 32λ���ݶ�
[SECTION .d32]
ALIGN	32
[BITS 32]
LABEL_SEG_DATA32:
times  4 db 0

SEG_DATA32_LEN	equ $ - LABEL_SEG_DATA32


