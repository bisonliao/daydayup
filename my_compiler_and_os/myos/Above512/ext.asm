;�����ļ��������.bin�ļ�д�����̵ĵ�2��������Ϊext���ֱ�pm.bin����

%if 0
;�������ʵģʽ���������ģ�����������Ҫ
;���򣬲���Ҫ����Ϊ����ģʽ�£�gs�Ѿ�����ʼ��Ϊ�ʵ���ѡ������
[BITS 16]
	mov ax, 0B800h
	mov gs, ax
%else

[BITS 32]

%endif
	mov edi, (80*10+0)*2



	mov  al, 'A'
	mov ah, 0ch			 ;��ʾ����ɫ
	mov [gs:edi], ax	 ;��ʾ
	jmp $
