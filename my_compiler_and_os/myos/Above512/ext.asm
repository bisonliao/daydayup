;将该文件编译出的.bin文件写入软盘的第2扇区，作为ext部分被pm.bin加载

%if 0
;如果是在实模式下跳过来的，下面两行需要
;否则，不需要，因为保护模式下，gs已经被初始化为适当的选择子了
[BITS 16]
	mov ax, 0B800h
	mov gs, ax
%else

[BITS 32]

%endif
	mov edi, (80*10+0)*2



	mov  al, 'A'
	mov ah, 0ch			 ;显示的颜色
	mov [gs:edi], ax	 ;显示
	jmp $
