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
.globl use_fpu
.type use_fpu, @function
use_fpu:
    pushl %ebp
    movl %esp, %ebp

    subl $8, %esp

    fldl a
    fldl b
    fdivp



    movl %ebp, %esp
    popl %ebp
    ret
