#include "const_def.h"
#include "time.h"
#include "fs.h"

#include "global.h"
#include "redefine.h"

typedef uint32_t (*SYSCALL_FUNC_PTR)(uint32_t,uint32_t,uint32_t,uint32_t, uint32_t);
static int kill_proc(int pid);


uint32_t       execute_sys_call(uint32_t eax,
		uint32_t ebx,
		uint32_t ecx,
		uint32_t edx, uint32_t ds)
{
	SYSCALL_FUNC_PTR func;	
	func = NULL;
	if (eax >= g_syscall_nr)
	{
		return -1;
	}

	func = (SYSCALL_FUNC_PTR)(g_syscall_entry[eax]);
	return  func(eax, ebx, ecx, edx, ds);
}
uint32_t   sys_time(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	return current_time();
}
uint32_t   sys_cin(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
// int cin(char *buf, int len);
	uint32_t addr;	
	unsigned int i;
	char * buf;
    char c;

    if (ebx == 0 && ecx < 0) return -1;

	//user_space_vaddr_to_paddr( (TProcess *)g_current, ebx, &addr, ds);
    addr = ebx;
	buf = (char * )addr;
	for (i = 0; i < ecx; ++i)
	{
        keyboard_read(&c);
        buf[i] = c;
        if (c == '\n')
        {
            return i+1;
        }
	}
	return i;
}
uint32_t   sys_cout(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	// count(const char * buf, int len);
	uint32_t addr;	
	unsigned int i;
	char * buf;
	//printk("start cout...\n");
	//user_space_vaddr_to_paddr( (TProcess *)g_current, ebx, &addr, ds);
    addr = ebx;
	buf = (char * )addr;
    buf = (char*)ebx;
	for (i = 0; i < ecx; ++i)
	{
		print_chr(buf[i]);
	}
	return 0;
}

uint32_t   sys_get_ticks_lo(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	return g_ticks & 0xffffffff;
}
uint32_t   sys_get_ticks_hi(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	return (g_ticks >> 32) & 0xffffffff;
}
uint32_t   sys_sleep(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	//printk("in syscall sleep(), modifying status to %d\n", PROC_STATUS_SLEEPING);
	g_current->alarm = g_ticks + ebx; /*从现在起 ebx个ticks后唤醒*/
	g_current->status = PROC_STATUS_SLEEPING;
	schedule();
	return 0;
}

/* ssize_t hd(uint32_t abs_sector, void * buf, uint32_t cmd); */
uint32_t   sys_hd(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	uint32_t addr;
	TBuffer * bh = NULL;
	if (ecx == 0 || (edx != WIN_READ && edx != WIN_WRITE))
	{
		return 0;
	}
    /*
	user_space_vaddr_to_paddr( (TProcess *)g_current, ecx, &addr, ds);
    */
    addr = ecx;
	if (edx == WIN_READ )
	{
		bh = buffer_read( ebx);
		memcpy( (void*)addr, (const void*)bh->data, 512);
		buffer_release(bh);
	}
	else
	{
		TBuffer * bh = buffer_lock( ebx);
		memcpy((void*)bh->data, (const void*)addr, 512);
		bh->flags = bh->flags | BUFFER_FLAG_DIRTY;
		buffer_release(bh);
	}
	return 0;
}
uint32_t   sys_sync(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	buffer_sync();
}
uint32_t   sys_test(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
#if 1
	free_inode(ebx);
	return 0;
#else
	return alloc_inode();
#endif
}
/** 从文件系统中把可执行程序拷贝到内存里，为运行作准备*/
int load_proc_from_fs(const char * pathname, int pid)
{
	int pathname_len;
	int iret;
	struct m_inode * p_inode;
	uint32_t offset = 0;
	int len;
	char buf[512]; /*内核态栈太小了，放到全局区可以不? 应该不可以，如果多个进程都在用呢?*/
	uint32_t user_space_org;
	uint32_t inode_nr;
	TProcess * p_proc = NULL;
	TBuffer * bh = NULL;

	/*检查文件*/
	pathname_len = strlen(pathname);

	iret =  namei(pathname, pathname_len, &inode_nr);
	if (iret != 0)
	{
		printk("namei('%s'... failed!", pathname);
		return -1;
	}
	p_inode = iget(inode_nr, &bh);
	if (p_inode == NULL)
	{
		printk("iget(%d... failed!", inode_nr);
		return -1;
	}
	if (p_inode->i_type != FILE_TYPE_REGULAR)
	{
		buffer_release(bh);
		printk("invalid file %s", pathname);
		return -1;
	}
	buffer_release(bh);

	//printk("%s %d: pid=%d\n", __FILE__, __LINE__, pid);
	user_space_org = FIRST_PROC_ORG+ (pid-1)*PROC_SPACE;	/*进程空间的物理内存开始位置*/


	/*将磁盘中的文件载入物理内存*/
	while (1)
	{
		len =  file_read(inode_nr,  offset, sizeof(buf), buf);
		if (len < 0)
		{
			printk("file_read return %d", len);
			return -1;
		}
		if (len == 0)
		{
			if (offset == 0)
			{
				printk("file length is zero!");
				return -1;
			}
			break;
		}
		if (offset + len > (PROC_SPACE-1000) ) /*用户空间栈预留1000字节*/
		{
			printk("file too long!\n");
			return -1;
		}
		memcpy( (void *)(user_space_org+offset), buf, len);
		offset += len;
	}

}
int sys_fork(uint32_t eip, uint32_t cs, uint32_t eflags, uint32_t esp, uint32_t ss, 
		uint32_t edi, uint32_t esi, uint32_t ebp, uint32_t edx, uint32_t ecx, uint32_t ebx, uint32_t eax)
{
	int  i;
	int pid;
	uint32_t user_space_org;
	TProcess * p_proc = NULL;


	for (pid = 1; pid < MAX_PROC_NR; ++pid)
	{
		if (g_procs[pid].pid == 0)
		{
			break;
		}
	}
	if (pid >= MAX_PROC_NR)
	{
		printk("no free entry in g_procs!\n");
		return -1;
	}
    /* 使用内核的paging，即线性地址等于物理地址*/
    _set_cr3(0);

	//printk("run to %s %d\n", __FILE__, __LINE__);

	user_space_org = FIRST_PROC_ORG+ (pid-1)*PROC_SPACE;	/*目标进程空间的物理内存开始位置*/
	/*拷贝用户态进程空间*/
	memcpy( (void *)(user_space_org), 
			(void *)(FIRST_PROC_ORG+ (g_current->pid-1)*PROC_SPACE),
			PROC_SPACE);
	//printk("run to %s %d\n", __FILE__, __LINE__);
    /*恢复进程自己的paging */
    _set_cr3( get_cr3_for_proc(g_current->pid) );

    

	/*设置进程控制数据结构*/
	p_proc = &g_procs[pid];
	p_proc->ldt_sel = g_first_ldt_selector + 8*pid;

    /*ldt直接映射4G全空间，相当于跳过段式管理*/
    /*用户态代码段*/
	init_descriptor( (TDescriptor*)&p_proc->ldts[0], 0, 0xfffff, 
			DA_C|DA_32|DA_PAGE| ( PRIVILEGE_USER << 5));
    /*用户态数据段、堆栈段*/
	init_descriptor( (TDescriptor*)&p_proc->ldts[1], 0, 0xfffff, 
			DA_DRW|DA_32|DA_PAGE| ( PRIVILEGE_USER << 5));

    /*描述进程的ldt表所在内存段的GDT表项*/
	init_descriptor( &g_gdt[ p_proc->ldt_sel >> 3 ], 
			(uint32_t)p_proc->ldts, 
			sizeof(TDescriptor)*MAX_LDT_ENT_NR-1, 
			DA_LDT);
    /*描述进程的tss段的GDT表项*/
	init_descriptor(&g_gdt[(g_first_tss_selector+8*pid) >> 3],
			(uint32_t)&p_proc->tss,
			sizeof(p_proc->tss) - 1,
			DA_386TSS);

	p_proc->cwd_inode = g_current->cwd_inode; /*当前工作目录的inode*/
	p_proc->root_inode = g_current->root_inode; /*根目录的inode*/
	for (i = 0; i < MAX_FD_NR; ++i)
	{
		p_proc->fd[i] = g_current->fd[i]; /*打开文件句柄，是file table数组下标*/
		if (p_proc->fd[i])
		{
			++(g_file_table[ p_proc->fd[i] ].f_count);
		}
	}

	/*子进程开始执行的位置就是fork系统调用返回那一刻的位置*/
	p_proc->pid			= pid;
	p_proc->tss.ss0		= g_kernel_data_selector;
	p_proc->tss.esp0		= (uint32_t)GET_KRNL_STACK_START(p_proc->pid);
    /*伪造几个选择子，RPL_USER表示是从用户态进入的 */
    /*SA_TIL属性表示该选择子是一个ldt的选择子，需要参考ldtr寄存器里的内容来找到ldt*/
	p_proc->tss.cs     = ((8 * 0) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.ds     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.es     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.fs     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.ss     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.gs     = p_proc->tss.ds;
	p_proc->tss.ldt     = p_proc->ldt_sel;
    /* paging :) */
    if (setup_paging_for_proc(pid) != 0) { p_proc->pid = 0; return -2;}
	p_proc->tss.cr3     = get_cr3_for_proc(pid);
	p_proc->tss.eip    = eip;
	p_proc->tss.esp    = esp;
	p_proc->tss.eax    = 0;	/*子进程返回0*/
	p_proc->tss.ebx    = ebx;
	p_proc->tss.ecx    = ecx;
	p_proc->tss.edx    = edx;
	p_proc->tss.ebp    = ebp;
	p_proc->tss.esi    = esi;
	p_proc->tss.edi    = edi;
	p_proc->tss.eflags = eflags;
	p_proc->status		= PROC_STATUS_RUNNING;
	p_proc->nice		= 35;
	p_proc->counter		= 0xff - p_proc->nice;
	p_proc->ppid		= g_current->pid; /*父子关系*/
	//printk("run to %s %d\n", __FILE__, __LINE__);

    /*父进程的一些属性也可能需要修改*/

	return pid;
}
int sys_exec(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds, 
    uint32_t * eip_for_userspace,
    uint32_t * esp_for_userspace)
{
/* int exec(const char * filename, char * const argv[]); */
/*
 *  ebx是系统调用的第一个参数，这里是filename
 *  ecx是系统调用的第二个参数，这里是argv
 */
	int  iret, i;
	int pid;
	uint32_t user_space_org, addr, stack_bottom;
    char  filename[255];
    uint32_t* argv_from_exec;
    char * *argv = NULL;
    int argc = 0;
    char * str;
	TProcess * p_proc = NULL;

    /***************************************************/
    /*建立用户态空间*/

    pid = g_current->pid;
	user_space_org = FIRST_PROC_ORG+ (pid-1)*PROC_SPACE;	/*进程用户态空间的物理内存开始位置*/
    strncpy(filename, (const char *)ebx, sizeof(filename)-1); 


    /*把参数都拷贝到用户态空间的尾部，即栈的底部*/
    if (ecx != 0) /*如果有参数, 肯定有，至少有第一个参数：可执行程序本身*/
    {
        int arglen = 0;
        argv_from_exec =  (uint32_t*)ecx;
        /*计算一下参数和argv指针需要多长*/
        for  (argc = 0; argv_from_exec[argc] != NULL; ++argc)
        {
            str = (char*)argv_from_exec[argc];
            arglen += strlen(str) + 1; /*1是末尾0的长度*/
         //   printk("argv[%d]=[%s]\n", argc, str);

            if (arglen > 1024 || argc > 9)
            {
                printk("%s %d: invalid arguments !\n", __FILE__, __LINE__);
                kill_proc(pid);
                schedule();
            }
        }
        if (argc < 1)
        {
            printk("%s %d: invalid arguments !\n", __FILE__, __LINE__);
            kill_proc(pid);
            schedule();
        }
        char * pCur =  USER_SPACE_VADDR_BOTTOM & 0xfffffffc;
        pCur -= arglen -  sizeof(char*)*(argc+1);
        stack_bottom = (uint32_t)pCur & 0xfffffffc; /*记住这个位置，稍后会从这个位置开始压栈*/

        argv = (char **)(pCur);
        pCur += sizeof(char*)*(argc+1);
        for  (i = 0; i < argc; ++i)
        {
            str = (char*)argv_from_exec[i];
            int len = strlen(str) + 1;
            memcpy(pCur, str, len);
            argv[i] =  pCur;
            pCur += len;
        }
        argv[argc] = NULL;
    }
    else
    {
        printk("%s %d: kill proc!\n", __FILE__, __LINE__);
        kill_proc(pid);
	    schedule();
    }
    //压入argc, argv
    if (argc == 0)
    {
        *(int *)(stack_bottom-4) = 0; /*压入NULL指针*/
    }
    else
    {
        *(int *)(stack_bottom-4) = argv;
    }
    *(uint32_t *)(stack_bottom-8) = argc; /*压入argc */
    stack_bottom -= 8;


    /*使用线性地址等于物理地址的分页映射*/
    _set_cr3(0);
    //printk("run to %s, %s, %d\n", __FUNCTION__, __FILE__, __LINE__);
    /*从文件系统里拷贝程序*/
    /*flat格式的可执行程序，bss不包括在文件中，但是会紧随data之后*/
    iret = load_proc_from_fs(filename, pid);
    /*恢复进程自己的paging */
    _set_cr3( get_cr3_for_proc(g_current->pid) );
    if (iret)
    {
        /*进程的用户态空间可能已经被破坏不完整了，终止该进程*/
        printk("%s %d: kill proc!\n", __FILE__, __LINE__);
        kill_proc(pid);
	    schedule();
    }

    /***************************************************/
    /*修改内核相关数据结构*/
	/*设置进程控制数据结构*/
	p_proc = &g_procs[pid];

    /*打开的文件要都关闭*/
	for (i = 0; i < MAX_FD_NR; ++i)
	{
		if (p_proc->fd[i])
        {
            int i_file;
            uint32_t inode;
            struct m_inode *p_inode;
            TBuffer * bh;
            i_file = g_current->fd[i];
            if (i_file == 0 || i_file >= MAX_FILE_TABLE)
            {
                panic(" file structure distroyed!\n");
                return -2;
            }
            if (g_file_table[i_file].f_count > 0)
            {
                --g_file_table[i_file].f_count;
            }
            if (g_file_table[i_file].f_count == 0)
            {
                inode = g_file_table[i_file].f_inode;
                memset(&(g_file_table[i_file]), 0, sizeof(g_file_table[i_file]));
            }
            else
            {
                continue;
            }


            p_inode = iget(inode, &bh); 
            if (p_inode == NULL)
            {
                return -4;
            }
            if (p_inode->i_count > 0)
            {
                p_inode->i_count--;
            }

            bh->flags |= BUFFER_FLAG_DIRTY;
            buffer_release(bh);
        }
	}
    //printk("run to %s, %s, %d\n", __FUNCTION__, __FILE__, __LINE__);


    /*用户态的栈指针和指令指针需要修改一下， 当exec系统调用返回的时候就生效了*/
    *eip_for_userspace = USER_SPACE_VADDR_HEAD;
    /*用户态堆栈被清空, esp指向stack_bottom位置*/
    *esp_for_userspace = stack_bottom;

	p_proc->tss.eax    = 0;	/*进程返回0, 不会被用到*/
	p_proc->status		= PROC_STATUS_RUNNING;
	p_proc->nice		= 35;
	p_proc->counter		= 0xff - p_proc->nice;
    //printk("run to %s, %s, %d\n", __FUNCTION__, __FILE__, __LINE__);
	return pid;
}
static int kill_proc(int pid)
{
    uint32_t user_space_org;
    int i;
    TProcess * p_proc = NULL;

	p_proc = &g_procs[pid];
    if (p_proc->status == PROC_STATUS_WAITING)
    {
        printk("%s %d:this proc can NOT be killed, may be it is in \n"
              "one of the implicit lists of the kernel.\n", __FILE__, __LINE__);
        return 100; /*特殊返回值，注意保持，表示这个进程暂时杀不得*/
    }

    _set_cr3(0);

	user_space_org = FIRST_PROC_ORG+ (pid-1)*PROC_SPACE;	/*进程空间的物理内存开始位置*/
    memset(user_space_org, 0, PROC_SPACE);



    /****************************************************/
    /*释放占用的系统资源*/
    /*1.打开的文件要都关闭*/
	for (i = 0; i < MAX_FD_NR; ++i)
	{
		if (p_proc->fd[i])
        {
            int i_file;
            uint32_t inode;
            struct m_inode *p_inode;
            TBuffer * bh;
            i_file = g_current->fd[i];
            if (i_file == 0 || i_file >= MAX_FILE_TABLE)
            {
                printk("%s %d: i_file invalid!\n", __FILE__, __LINE__);
                continue;
            }
            if (g_file_table[i_file].f_count > 0)
            {
                --g_file_table[i_file].f_count;
            }
            if (g_file_table[i_file].f_count == 0)
            {
                inode = g_file_table[i_file].f_inode;
                memset(&(g_file_table[i_file]), 0, sizeof(g_file_table[i_file]));
            }
            else
            {
                continue;
            }


            p_inode = iget(inode, &bh); 
            if (p_inode == NULL)
            {
                panic("%s %d: iget failed!\n", __FILE__, __LINE__);
                return -4;
            }
            if (p_inode->i_count > 0)
            {
                p_inode->i_count--;
            }

            bh->flags |= BUFFER_FLAG_DIRTY;
            buffer_release(bh);
        }
	}
    /*2.todo: 用户态内存如果是动态分配的, 需要释放*/
    /*3.所有子进程，父子关系需要转移*/
    for (i = 1; i < MAX_PROC_NR; ++i)
    {
        if (g_procs[i].pid > 0 &&
            g_procs[i].pid != pid &&
            g_procs[i].ppid == pid)
        {
            g_procs[i].ppid = 0; /*进程0会负责收尸*/
        }
    }


    p_proc->status = PROC_STATUS_ZOMBIE; /*进程状态为僵尸状态等待回收*/
	return 0;
}
uint32_t sys_exit(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
/* int exit();*/
	int  iret, i;
	int pid, ppid;
	uint32_t user_space_org, addr;

    pid = g_current->pid;
    ppid = g_current->ppid;

    //printk("ppid=%d\n", ppid);

    /*看看父进程是否正在等待子进程退出*/
    if (ppid > 0)
    {
        if (g_procs[ppid].pid != ppid)
        {
            panic("mismatch!\n");
        }
        if (g_procs[ppid].status == PROC_STATUS_SLEEPING &&
            g_procs[ppid].flag & PROC_FLAG_WAIT_CHILD)
        {
            //printk("modify parent(pid=%d) to running...\n", ppid);
            g_procs[ppid].status = PROC_STATUS_RUNNING;
            g_procs[ppid].flag &= ~(PROC_FLAG_WAIT_CHILD);
        }
    }

	return kill_proc(pid);
}
uint32_t sys_wait(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
    /* int wait(int pid, int * status) */
	uint32_t addr;
    uint32_t child_pid = ebx, pid;
    int * status = NULL, i;
    char has_child_flag = 0;

    pid = g_current->pid;

    if (ecx != 0)
    {
        /*
	    user_space_vaddr_to_paddr( (TProcess *)g_current, ecx, &addr, ds);
        */
        addr = ecx;
        status = (int *)addr;
    }
    //printk("%d wait %d\n", pid, child_pid);
again:
    if (child_pid > 0 && child_pid < MAX_PROC_NR) /*指定要等待某个子进程*/
    {
        if (g_procs[child_pid].pid == 0) /* no exist*/
        {
            return -10;
        }
        else if (g_procs[child_pid].pid == child_pid)
        {
            if (g_procs[child_pid].ppid != pid) /* not my child */
            {
                return -20;
            }
            if (g_procs[child_pid].status == PROC_STATUS_ZOMBIE)
            {
                /*给status赋值*/
                if (status) { status = g_procs[child_pid].exit_code;  };

                memset(&g_procs[child_pid], 0, sizeof(TProcess) );
                return child_pid;
            }
            /*睡一觉再看, 我儿子退出的时候会叫醒我的*/
            //printk("proc#%d go to sleep...\n", pid);
            g_current->status = PROC_STATUS_SLEEPING;
            g_current->alarm = 0;
            g_current->flag |= PROC_FLAG_WAIT_CHILD;
            schedule();
            goto again;
        }
        else 
        {
            panic("mismatch!\n");
        }
            
    }
    else  /*没有指定要等待某个子进程*/
    {
        for (i = 1; i < MAX_PROC_NR; ++i)/*遍历进程控制表*/
        {
            if (g_procs[i].pid > 0 &&
                    g_procs[i].ppid == pid )
            {
                // 找到了一个子进程
                has_child_flag = 1;

                if (g_procs[i].status == PROC_STATUS_ZOMBIE)
                {
                    //有子进程退出，并且是在等待任一子进程
                    /*给status赋值*/
                    if (status) { status = g_procs[child_pid].exit_code;  };

                    memset(&g_procs[i], 0, sizeof(TProcess) );
                    return i;
                }
            }
        }
        if (has_child_flag == 0)
        {
            return -30; /*没有子进程*/
        }
        //有子进程，但是都没有退出
        /*睡一觉再看, 我儿子退出的时候会叫醒我的*/
        g_current->status = PROC_STATUS_SLEEPING;
        g_current->alarm = 0;
        g_current->flag |= PROC_FLAG_WAIT_CHILD;
        schedule();
        goto again;
    }
    panic("this line should NOT be printed!\n");
    return 0;
}
