#include "api.h"
#include "types.h"
#include "global.h"
#include "const_def.h"


int main(int argc, char ** argv)
{
    int i;
    char buf[1];

    int len ;

    printf("\n welcome to %s :)\n", argv[0]);

    for (i = 0; i < 4; ++i)
    {
        len = _cin(buf, 1);
        if (len == 1)
        {
            printf("{%d}", buf[0]);
        }
    }
    use_fpu();

    printf("\n%s exit... \n", argv[0], i);

    return -1;
}
