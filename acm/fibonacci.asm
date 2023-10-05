; ��x86_64���дһ��fibonacci����
; ��Ϥx86_64���ĺ������ù���

section .text
global fibo
fibo:
    push rbp
    mov  rbp, rsp
    ; ������ʱ�õ��������Ĵ��������ڻָ�
    push rbx
    push rcx
    sub  rsp, 24 ; �ֲ������Ŀռ�

    ; rdi  #1 argument
    ; �����������1
    cmp rdi, 1
    jne  check_2
    mov  rax, 1
    jmp go_label

check_2:
    ; �����������2
    cmp rdi, 2
    jne  check_other_value
    mov  rax, 1
    jmp go_label

check_other_value:
    mov [rbp-24], rdi  ; ��������������ֲ�����

    mov rdi, [rbp-24]   ; �Ӿֲ������ָ����������
    sub rdi, 1
    call fibo
    mov rbx, rax   ; ������浽rbx

    mov rdi, [rbp-24]
    sub rdi, 2
    call fibo
    add rax, rbx ;  f(x-1) + f(x-2)

go_label:
    add  rsp, 24
    pop  rcx
    pop  rbx
    mov  rsp, rbp
    pop  rbp
    ret

