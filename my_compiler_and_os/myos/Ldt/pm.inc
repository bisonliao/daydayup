
%macro Descriptor 3
	dw %2 & 0ffffh
	dw %1 & 0ffffh
	db (%1 >> 16) & 0ffh
	dw ((%2 >> 8)& 0f00h) | (%3 & 0f0ffh)
	db (%1 >> 24) & 0ffh
%endmacro

DA_32		EQU	4000h	; 32 位段
DA_DRW		EQU	92h	; 存在的可读写数据段属性值
DA_C		EQU	98h	; 存在的只执行代码段属性值
DA_LDT		EQU	  82h	; 局部描述符表段类型值
