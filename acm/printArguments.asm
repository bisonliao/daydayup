; 为了熟悉x86_64汇编， 练习写代码打印命令行参数
section .data
msg  db   10, 0 ; newline
yes  db    "yes", 10, 0 ;debug msg

section .text
global main
main:
;  rdi -- argc
;  rsi -- argv
    push rbp
    mov  rbp, rsp

    sub  rsp, 24  ; 3 local variables, not used actually

    push rbx
    push rdi
    push rsi

    mov  rax, 1 ;return value
    mov  rbx, rdi ; rbx for count, decrease while print arguments, do NOT use rcx, it is special

    cmp  rbx, 0
    jle  main_end

printAgain:
    mov  rdi, [rsi];  print argument
    call printStr

    mov  rdi, msg; print newe line
    call printStr

    add  rsi, 8 ; shift to the next argument
    dec  rbx ;decrease count
    cmp  rbx, 0
    jne  printAgain

    mov  rax, 0 ; return value of main function
main_end:
    mov  rdi, yes ;print debug information
    call printStr

    pop  rsi
    pop  rdi
    pop  rbx

    mov  rsp, rbp
    pop  rbp
    ret


global printStr
printStr:
    push rbp
    mov  rbp, rsp
    sub  rsp, 24

    push rdi
    push rsi
    push rdx

    call strLen
    mov  [rbp-8], rax ; len of str
    mov  [rbp-16], rdi; str addr

    mov  rax, 1; SYS_write
    mov  rdi, 1; STDOUT
    mov  rsi, [rbp-16]; address of str to print
    mov  rdx, [rbp-8] ; len of str to print
    syscall

    pop  rdx
    pop  rsi
    pop  rdi

    mov  rsp, rbp
    pop  rbp
    ret

global strLen
strLen:
    push rbp
    mov  rbp, rsp

    push rdi

    mov  rax, 0
step:
    cmp  byte [rdi], 0
    je   l_end
    inc  rdi
    inc  rax
    jmp  step
l_end:
    pop  rdi

    mov  rsp, rbp
    pop  rbp
    ret

