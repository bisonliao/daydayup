#include "pool.h"

void func(string & s)
{
    printf("do task, %s\n", s.c_str());
}

int main()
{
    {
        ThreadPool tp(0);
        int cnt = 0;
        for (int i = 0; i < 30; ++i)
        {
            Task t;
            char buf[100];
            snprintf(buf, sizeof(buf), "arg:%d", cnt++);
            t.m_arg = string(buf);
            t.m_func = func;

            tp.push(t);
            //sleep(1);
        }
        
    }
    sleep(1);
  
    return 0;
}
