                .text
                .globl  _delay10us
_delay10us  :
        movl $1000, %eax
    delay_again:
        nop
        decl %eax
        cmpl $0, %eax
        jne delay_again
        ret


//****************************************************************
//*                                                              *
//* Calculate the IEEE address checksum                          *
//*                                                              *
//* From C:                                                      *
//*      caleepcrc(bytecount,x[n])                               *
//*                                                              *
//****************************************************************
                .globl  _caleepcrc
_caleepcrc      :
                ret
