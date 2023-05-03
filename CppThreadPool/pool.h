#ifndef POOL_H_INCLUDED
#define POOL_H_INCLUDED
#endif

#include <iostream>
#include <memory>
#include <thread>
#include <queue>
#include <string>
#include <mutex>
#include <unistd.h>
#include <condition_variable>
#include <stdio.h>
#include <stdlib.h>
#include <functional>
#include <atomic>

using namespace std;

class Task // task to do by thread
{
    public:
        std::function<void(string &)> m_func;
        string m_arg;
};



class ThreadPool
{
    private:
        queue<Task> m_q; // tasks to do 
        queue<std::thread*> m_t; // threads in pool
        mutex m_mutex; 
        condition_variable m_cv;
        std::atomic<bool> m_stopFlag; // stop flag for threads
    public:
        ThreadPool(int threadNum = 0):m_stopFlag(false)
        {
            if (threadNum == 0)
            {
                threadNum = std::thread::hardware_concurrency();
            }
            if (threadNum <= 1) { threadNum = 4;}
            for (int i = 0; i < threadNum; ++i)
            {
                thread *t = new thread(&ThreadPool::run, this);
                m_t.push(t);
            }
        }
        void push(const Task &task)
        {
            std::unique_lock<std::mutex> lk(m_mutex); 
            m_q.push(task);
            m_cv.notify_one();
            printf("task queue len:%d\n", m_q.size());
        }
        void run()
        {
            while(!m_stopFlag)
            {
                std::unique_lock<std::mutex> lk(this->m_mutex); 
                m_cv.wait(lk, [this]{return !this->m_q.empty() || this->m_stopFlag;}); 
                if (m_stopFlag)
                {
                    printf("thread stop!\n");
                    break;
                }
                Task task = this->m_q.front();
                this->m_q.pop();
                lk.unlock(); 
                task.m_func(task.m_arg);
            }
        }
        ~ThreadPool()
        {
            m_stopFlag = true;
            for (int i = 0; i < m_t.size(); ++i)
            {
                m_cv.notify_one();
            }
            m_cv.notify_all();

            
        }
        ThreadPool(ThreadPool &) = delete;
        ThreadPool(const ThreadPool &) = delete;
        ThreadPool & operator=(ThreadPool &) = delete;
        ThreadPool & operator=(const ThreadPool &) = delete;

};