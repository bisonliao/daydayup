; 用x86_64汇编写一个fibonacci函数
; 熟悉x86_64汇编的函数调用规则

section .text
global fibo
fibo:
    push rbp
    mov  rbp, rsp
    ; 保存临时用到的两个寄存器，用于恢复
    push rbx
    push rcx
    sub  rsp, 24 ; 局部变量的空间

    ; rdi  #1 argument
    ; 输入参数等于1
    cmp rdi, 1
    jne  check_2
    mov  rax, 1
    jmp go_label

check_2:
    ; 输入参数等于2
    cmp rdi, 2
    jne  check_other_value
    mov  rax, 1
    jmp go_label

check_other_value:
    mov [rbp-24], rdi  ; 保存输入参数到局部变量

    mov rdi, [rbp-24]   ; 从局部变量恢复出输入参数
    sub rdi, 1
    call fibo
    mov rbx, rax   ; 结果保存到rbx

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

