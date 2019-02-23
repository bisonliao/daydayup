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
	g_current->alarm = g_ticks + ebx; /*�������� ebx��ticks����*/
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
/** ���ļ�ϵͳ�аѿ�ִ�г��򿽱����ڴ��Ϊ������׼��*/
int load_proc_from_fs(const char * pathname, int pid)
{
	int pathname_len;
	int iret;
	struct m_inode * p_inode;
	uint32_t offset = 0;
	int len;
	char buf[512]; /*�ں�̬ջ̫С�ˣ��ŵ�ȫ�������Բ�? Ӧ�ò����ԣ����������̶�������?*/
	uint32_t user_space_org;
	uint32_t inode_nr;
	TProcess * p_proc = NULL;
	TBuffer * bh = NULL;

	/*����ļ�*/
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
	user_space_org = FIRST_PROC_ORG+ (pid-1)*PROC_SPACE;	/*���̿ռ�������ڴ濪ʼλ��*/


	/*�������е��ļ����������ڴ�*/
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
		if (offset + len > (PROC_SPACE-1000) ) /*�û��ռ�ջԤ��1000�ֽ�*/
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
    /* ʹ���ں˵�paging�������Ե�ַ���������ַ*/
    _set_cr3(0);

	//printk("run to %s %d\n", __FILE__, __LINE__);

	user_space_org = FIRST_PROC_ORG+ (pid-1)*PROC_SPACE;	/*Ŀ����̿ռ�������ڴ濪ʼλ��*/
	/*�����û�̬���̿ռ�*/
	memcpy( (void *)(user_space_org), 
			(void *)(FIRST_PROC_ORG+ (g_current->pid-1)*PROC_SPACE),
			PROC_SPACE);
	//printk("run to %s %d\n", __FILE__, __LINE__);
    /*�ָ������Լ���paging */
    _set_cr3( get_cr3_for_proc(g_current->pid) );

    

	/*���ý��̿������ݽṹ*/
	p_proc = &g_procs[pid];
	p_proc->ldt_sel = g_first_ldt_selector + 8*pid;

    /*ldtֱ��ӳ��4Gȫ�ռ䣬�൱��������ʽ����*/
    /*�û�̬�����*/
	init_descriptor( (TDescriptor*)&p_proc->ldts[0], 0, 0xfffff, 
			DA_C|DA_32|DA_PAGE| ( PRIVILEGE_USER << 5));
    /*�û�̬���ݶΡ���ջ��*/
	init_descriptor( (TDescriptor*)&p_proc->ldts[1], 0, 0xfffff, 
			DA_DRW|DA_32|DA_PAGE| ( PRIVILEGE_USER << 5));

    /*�������̵�ldt�������ڴ�ε�GDT����*/
	init_descriptor( &g_gdt[ p_proc->ldt_sel >> 3 ], 
			(uint32_t)p_proc->ldts, 
			sizeof(TDescriptor)*MAX_LDT_ENT_NR-1, 
			DA_LDT);
    /*�������̵�tss�ε�GDT����*/
	init_descriptor(&g_gdt[(g_first_tss_selector+8*pid) >> 3],
			(uint32_t)&p_proc->tss,
			sizeof(p_proc->tss) - 1,
			DA_386TSS);

	p_proc->cwd_inode = g_current->cwd_inode; /*��ǰ����Ŀ¼��inode*/
	p_proc->root_inode = g_current->root_inode; /*��Ŀ¼��inode*/
	for (i = 0; i < MAX_FD_NR; ++i)
	{
		p_proc->fd[i] = g_current->fd[i]; /*���ļ��������file table�����±�*/
		if (p_proc->fd[i])
		{
			++(g_file_table[ p_proc->fd[i] ].f_count);
		}
	}

	/*�ӽ��̿�ʼִ�е�λ�þ���forkϵͳ���÷�����һ�̵�λ��*/
	p_proc->pid			= pid;
	p_proc->tss.ss0		= g_kernel_data_selector;
	p_proc->tss.esp0		= (uint32_t)GET_KRNL_STACK_START(p_proc->pid);
    /*α�켸��ѡ���ӣ�RPL_USER��ʾ�Ǵ��û�̬����� */
    /*SA_TIL���Ա�ʾ��ѡ������һ��ldt��ѡ���ӣ���Ҫ�ο�ldtr�Ĵ�������������ҵ�ldt*/
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
	p_proc->tss.eax    = 0;	/*�ӽ��̷���0*/
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
	p_proc->ppid		= g_current->pid; /*���ӹ�ϵ*/
	//printk("run to %s %d\n", __FILE__, __LINE__);

    /*�����̵�һЩ����Ҳ������Ҫ�޸�*/

	return pid;
}
int sys_exec(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds, 
    uint32_t * eip_for_userspace,
    uint32_t * esp_for_userspace)
{
/* int exec(const char * filename, char * const argv[]); */
/*
 *  ebx��ϵͳ���õĵ�һ��������������filename
 *  ecx��ϵͳ���õĵڶ���������������argv
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
    /*�����û�̬�ռ�*/

    pid = g_current->pid;
	user_space_org = FIRST_PROC_ORG+ (pid-1)*PROC_SPACE;	/*�����û�̬�ռ�������ڴ濪ʼλ��*/
    strncpy(filename, (const char *)ebx, sizeof(filename)-1); 


    /*�Ѳ������������û�̬�ռ��β������ջ�ĵײ�*/
    if (ecx != 0) /*����в���, �϶��У������е�һ����������ִ�г�����*/
    {
        int arglen = 0;
        argv_from_exec =  (uint32_t*)ecx;
        /*����һ�²�����argvָ����Ҫ�೤*/
        for  (argc = 0; argv_from_exec[argc] != NULL; ++argc)
        {
            str = (char*)argv_from_exec[argc];
            arglen += strlen(str) + 1; /*1��ĩβ0�ĳ���*/
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
        stack_bottom = (uint32_t)pCur & 0xfffffffc; /*��ס���λ�ã��Ժ������λ�ÿ�ʼѹջ*/

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
    //ѹ��argc, argv
    if (argc == 0)
    {
        *(int *)(stack_bottom-4) = 0; /*ѹ��NULLָ��*/
    }
    else
    {
        *(int *)(stack_bottom-4) = argv;
    }
    *(uint32_t *)(stack_bottom-8) = argc; /*ѹ��argc */
    stack_bottom -= 8;


    /*ʹ�����Ե�ַ���������ַ�ķ�ҳӳ��*/
    _set_cr3(0);
    //printk("run to %s, %s, %d\n", __FUNCTION__, __FILE__, __LINE__);
    /*���ļ�ϵͳ��������*/
    /*flat��ʽ�Ŀ�ִ�г���bss���������ļ��У����ǻ����data֮��*/
    iret = load_proc_from_fs(filename, pid);
    /*�ָ������Լ���paging */
    _set_cr3( get_cr3_for_proc(g_current->pid) );
    if (iret)
    {
        /*���̵��û�̬�ռ�����Ѿ����ƻ��������ˣ���ֹ�ý���*/
        printk("%s %d: kill proc!\n", __FILE__, __LINE__);
        kill_proc(pid);
	    schedule();
    }

    /***************************************************/
    /*�޸��ں�������ݽṹ*/
	/*���ý��̿������ݽṹ*/
	p_proc = &g_procs[pid];

    /*�򿪵��ļ�Ҫ���ر�*/
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


    /*�û�̬��ջָ���ָ��ָ����Ҫ�޸�һ�£� ��execϵͳ���÷��ص�ʱ�����Ч��*/
    *eip_for_userspace = USER_SPACE_VADDR_HEAD;
    /*�û�̬��ջ�����, espָ��stack_bottomλ��*/
    *esp_for_userspace = stack_bottom;

	p_proc->tss.eax    = 0;	/*���̷���0, ���ᱻ�õ�*/
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
        return 100; /*���ⷵ��ֵ��ע�Ᵽ�֣���ʾ���������ʱɱ����*/
    }

    _set_cr3(0);

	user_space_org = FIRST_PROC_ORG+ (pid-1)*PROC_SPACE;	/*���̿ռ�������ڴ濪ʼλ��*/
    memset(user_space_org, 0, PROC_SPACE);



    /****************************************************/
    /*�ͷ�ռ�õ�ϵͳ��Դ*/
    /*1.�򿪵��ļ�Ҫ���ر�*/
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
    /*2.todo: �û�̬�ڴ�����Ƕ�̬�����, ��Ҫ�ͷ�*/
    /*3.�����ӽ��̣����ӹ�ϵ��Ҫת��*/
    for (i = 1; i < MAX_PROC_NR; ++i)
    {
        if (g_procs[i].pid > 0 &&
            g_procs[i].pid != pid &&
            g_procs[i].ppid == pid)
        {
            g_procs[i].ppid = 0; /*����0�Ḻ����ʬ*/
        }
    }


    p_proc->status = PROC_STATUS_ZOMBIE; /*����״̬Ϊ��ʬ״̬�ȴ�����*/
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

    /*�����������Ƿ����ڵȴ��ӽ����˳�*/
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
    if (child_pid > 0 && child_pid < MAX_PROC_NR) /*ָ��Ҫ�ȴ�ĳ���ӽ���*/
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
                /*��status��ֵ*/
                if (status) { status = g_procs[child_pid].exit_code;  };

                memset(&g_procs[child_pid], 0, sizeof(TProcess) );
                return child_pid;
            }
            /*˯һ���ٿ�, �Ҷ����˳���ʱ�������ҵ�*/
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
    else  /*û��ָ��Ҫ�ȴ�ĳ���ӽ���*/
    {
        for (i = 1; i < MAX_PROC_NR; ++i)/*�������̿��Ʊ�*/
        {
            if (g_procs[i].pid > 0 &&
                    g_procs[i].ppid == pid )
            {
                // �ҵ���һ���ӽ���
                has_child_flag = 1;

                if (g_procs[i].status == PROC_STATUS_ZOMBIE)
                {
                    //���ӽ����˳����������ڵȴ���һ�ӽ���
                    /*��status��ֵ*/
                    if (status) { status = g_procs[child_pid].exit_code;  };

                    memset(&g_procs[i], 0, sizeof(TProcess) );
                    return i;
                }
            }
        }
        if (has_child_flag == 0)
        {
            return -30; /*û���ӽ���*/
        }
        //���ӽ��̣����Ƕ�û���˳�
        /*˯һ���ٿ�, �Ҷ����˳���ʱ�������ҵ�*/
        g_current->status = PROC_STATUS_SLEEPING;
        g_current->alarm = 0;
        g_current->flag |= PROC_FLAG_WAIT_CHILD;
        schedule();
        goto again;
    }
    panic("this line should NOT be printed!\n");
    return 0;
}
