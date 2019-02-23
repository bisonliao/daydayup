#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "struct.h"

int main()
{
    printf("#ifndef OFFSET_H_INCLUDED\n #define OFFSET_H_INCLUDED\n");


    printf("#define fpuss_OFFSET_IN_TProcess %d\n", offset(TProcess, fpuss));
    printf("#define tss_OFFSET_IN_TProcess %d\n", offset(TProcess, tss));
    printf("#define tss_sel_OFFSET_IN_TProcess %d\n", offset(TProcess, tss_sel));
    printf("#define ldt_sel_OFFSET_IN_TProcess %d\n", offset(TProcess, ldt_sel));
    printf("#define ldts_OFFSET_IN_TProcess %d\n", offset(TProcess, ldts));
    printf("#define pid_OFFSET_IN_TProcess %d\n", offset(TProcess, pid));
    printf("#define alarm_OFFSET_IN_TProcess %d\n", offset(TProcess, alarm));
    printf("#define status_OFFSET_IN_TProcess %d\n", offset(TProcess, status));
    printf("#define counter_OFFSET_IN_TProcess %d\n", offset(TProcess, counter));
    printf("#define nice_OFFSET_IN_TProcess %d\n", offset(TProcess, nice));
    printf("#define cwd_inode_OFFSET_IN_TProcess %d\n", offset(TProcess, cwd_inode));
    printf("#define root_inode_OFFSET_IN_TProcess %d\n", offset(TProcess, root_inode));
    printf("#define fd_OFFSET_IN_TProcess %d\n", offset(TProcess, fd));
    printf("#define ppid_OFFSET_IN_TProcess %d\n", offset(TProcess, ppid));
    printf("#define flag_OFFSET_IN_TProcess %d\n", offset(TProcess, flag));
    printf("#define exit_code_OFFSET_IN_TProcess %d\n", offset(TProcess, exit_code));


    printf("#endif\n");
    return 0;
}
