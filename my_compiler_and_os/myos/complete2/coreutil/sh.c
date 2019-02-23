#include "api.h"
#include "types.h"
#include "global.h"
#include "redefine.h"
#include "const_def.h"

char buf[9000]= { 'A'};


static int test_float()
{
    char buf[100];
    double a, b, res;
    int fraction, integral;
    int len   ;
    char operator;
    printf("input first float number:");
    len = _cin(buf, sizeof(buf));
    if (len < 1) {return -1;}
    if (buf[len-1] == '\n')
    {
        buf[len-1] = '\0';
    }
    a = strtod(buf, NULL);
    //printf("a=%d.%d\n", (int)a,  (int)((a-(int)a) * 1000000));

    printf("input operator:");
    len = _cin(buf, sizeof(buf));
    if (len < 1) {return -1;}
    operator = buf[0];

    printf("input second float number:");
    len = _cin(buf, sizeof(buf));
    if (len < 1) {return -1;}
    if (buf[len-1] == '\n')
    {
        buf[len-1] = '\0';
    }
    b = strtod(buf, NULL);
    //printf("a=%d.%d\n", (int)b,  (int)((b-(int)b) * 1000000));

    switch (operator)
    {
        case '+': 
            res = a + b;
            break;
        case '-': 
            res = a - b;
            break;
        case '*': 
            res = a * b;
            break;
        case '/': 
            res = a / b;
            break;
        default:
            printf("invalid input!\n");
            return -1;
    }
    integral = (int)res;
    fraction = (int)((res - integral) * 10000000);
    printf("%d.%d\n", integral, fraction);
    return 0;
}
int run_sh2()
{
    int i, pid;
    char argv0[] = "sh2";
    char argv1[] = "haha";
    char argv2[] = "hehe";
    char * argv[10];

    pid = _fork();

    if ( pid < 0)
    {
        return -1;
    }
    if (pid == 0)
    {
        argv[0] = argv0;
        argv[1] = argv1;
        argv[2] = argv2;
        argv[3] = NULL;
        i = _exec("/bin/sh2", argv);
        return 0;
    }
    else
    {
        _wait(pid, NULL);
    }
    return 0;
}

int main(int argc, char **vv)
{
	char buf2[512];
	int i;
    int len;


	printf("%s begin to run...\n", vv[0]);
    

    while (1)
    {
        printf("sh: ");

        len = _cin(buf2, sizeof(buf2));
        if (len < 1)
        {
            continue;
        }
        if (buf2[len-1] == '\n')
        {
            buf2[len-1] = '\0';
        }
        if (strlen(buf2) == 4 &&
            strncmp(buf2, "exit", 4) == 0)
        {
            break;
        }

        printf("your input is [%s]\n", buf2);

        if (strlen(buf2) == 3 &&
            strncmp(buf2, "sh2", 3) == 0)
        {
            run_sh2();
        }
        if (strlen(buf2) == 3 &&
            strncmp(buf2, "cls", 3) == 0)
        {
            for (i = 0; i < 32; ++i)
                printf("                                                           \n");
        }
        if (strlen(buf2) == 5 &&
            strncmp(buf2, "float", 5) == 0)
        {
            test_float();
            //i = use_fpu_and_switch();
            //printf("use_fpu_and_switch returned %d\n", i);
        }
    }



    return 0;
}

