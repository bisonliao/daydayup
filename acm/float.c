#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>


int main()
{

	uint64_t v;
	uint64_t count;

	if (sizeof(float)!=4)
	{
		return;
	}
	uint32_t max1 =  0x80000000;
	uint32_t max2 =  0xffffffff;


	//这段代码证明了两点：
	//1、对于float这样一个四字节的浮点数，内存两段单调递增：0xffffffff->0x80000000，和0x00000000->0x80000000
	//2、浮点数的绝对值越大， 两个连续确定状态之间的漏过的实数段就越大
	double prev_float = 0;
	count = 0;
	for (v = max2; v >= max1; --v)
	{
		count++;

		uint32_t vv = v;
		double f = *(float*)&vv;

		if ( (count % 1000000) == 0)
		{
			printf("0x%llx, %f, %f\n", v, f, f - prev_float );
		}

		if (v != max2)
		{
			if (f < prev_float)
			{
				printf("not increase always! %llu, 0x%llx, %f,%f\n", v, v, f, prev_float);
				return -1;
			}
		}
		prev_float = f;

	}

	for (v = 0; v <= max1; ++v)
	{
		count++;

		uint32_t vv = v;
		double f = *(float*)&vv;
		if ( (count % 1000000) == 0)
		{
			printf("0x%llx, %f, %f\n", v, f, f - prev_float );
		}

		if (v > 0)
		{
			if (f < prev_float)
			{
				printf("not increase always! %llu, 0x%llx, %f,%f\n", v, v, f, prev_float);
				return -1;
			}
		}
		prev_float = f;

	}


	//////////////////////////
	//这段代码，演示即使简单的加减运算，也会导致浮点数出现计算误差
	//因为两个浮点数相加，得出的结果，可能不在64bit内存有限的表示状态中，这时候就会出现“取整”误差
        double x = 1.0;
        uint64_t v;

        if (sizeof(double)!= 8)
        {
                return -1;
        }

        v = *(uint64_t*)&x;
        printf("0x%llx\n", v);

        unsigned int i;
        srand(time(NULL));
        for (i = 0; i < 40000; ++i)
        {
                double a = (double)rand()/RAND_MAX;
//              double b = (double)rand()/RAND_MAX;

                double a = (double)rand()/RAND_MAX;
//              double b = (double)rand()/RAND_MAX;

                x += a;
                x -= a;

                if (x!=1.0)
                {
                        v = *(uint64_t*)&x;
                        printf("0x%llx, %f\n", v, x);
                }
        }
	
	return 0;

}
