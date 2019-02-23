/*
 * double a = 12.3;
 * double b = 3.56;
 */
.data
.align 8
.type   a, @object
.size   a, 8
a:
.long   -1717986918
.long   1076402585

.align 8
.type   b, @object
.size   b, 8
b:
.long   1202590843
.long   1074559713

.align 4
.type c, @object
.size c, 4

.text
/*
 * use_fpu_and_switch函数计算12.3*3.56,然后结果作为整数返回
 */
.globl use_fpu_and_switch
.type use_fpu_and_switch, @function
use_fpu_and_switch:
    pushl %ebp
    movl %esp, %ebp

    subl $8, %esp

    fldl a
    fldl b
    fmulp

    call  run_sh2

    fistpl -4(%esp)
    movl -4(%esp), %eax

    movl %ebp, %esp
    popl %ebp
    ret
.globl use_fpu
.type use_fpu, @function
use_fpu:
    pushl %ebp
    movl %esp, %ebp

    subl $8, %esp

    fldl a
    fldl b
    fdiv



    movl %ebp, %esp
    popl %ebp
    ret
