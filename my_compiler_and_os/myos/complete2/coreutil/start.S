.text
.org 0x0 /*这一行为什么重要呢？ 
          * 表示它是text节的起始位置，避免把其他
          * 函数放到text节的第一个字节处。保证一
          * 进入0x4000000就能运行到_start*/
.globl _start
_start:

/**
 * 下面用jmp和call是等价的
 *  为什么不为main压入参数涅？
 *  因为内核在创建该进程的时候准备好了参数
 */
#if 0
    pushl $return_here
    jmp main
return_here:

#else
    call main
#endif

    pushl %eax
    call _exit

