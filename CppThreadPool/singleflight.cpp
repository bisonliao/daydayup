#include <mutex>
#include <map>
#include <future>
#include <string>
#include <thread>
#include <unistd.h>
#include <condition_variable>
using namespace std;


class CSingleFlight
{
    private:
        mutex m_mut;
        bool m_isFirst;//只执行一次
        condition_variable m_cv;
        bool m_ready;

    public:
        CSingleFlight():m_isFirst(true), m_ready(false)
        {

        }
        void run(string key, function<void(string&)>  f)
        {
            unique_lock<mutex> lk(m_mut);
            if (m_isFirst)
            {
                m_isFirst = false;
                f(key); m_ready = true; m_cv.notify_all();
            }
            else
            {
                m_cv.wait(lk, [this](){return this->m_ready;});
            }

            printf("thread run %d\n", m_ready);
        }

};
void doSomething(string & s)
{
    printf("do something, %s\n", s.c_str());
    sleep(1);
}
int main()
{
    CSingleFlight sf;
    int i;
    for (i = 0; i < 20; ++i)
    {
        thread t = thread(&CSingleFlight::run, &sf,  "hello",  function<void(string&)>(doSomething));
        t.detach();
    }
    while (1) {}
    return 0;

}
	    

