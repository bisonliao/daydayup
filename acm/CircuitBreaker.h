// 熔断器的实现，具体介绍可见：https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker
#ifndef CIRCUIT_BREAKER_H_INCLUDED
#define  CIRCUIT_BREAKER_H_INCLUDED

#include <map>
#include <string>
#include <stdint.h>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <random>

using namespace std;



enum CIRCUIT_BREAKER_STATE
{
    CLOSE = 0,
    OPEN = 1,
    HALF_OPEN = 2,
};

class CircuitBreakerConfig //熔断器的配置
{
public:
    uint32_t m_timeOutSec4Open; // open/half open状态下的超时时间 s
    uint32_t m_clearStatisticInterval; //清理计数器的时间间隔 s
    uint32_t m_minReqCount; // 统计信息有效的要求最少的请求个数，请求个数太少就不算有效
    float m_failThreshhold2OpenState; // 转open状态的失败阈值
    float m_failThreshhold2CloseState; // 转close状态的失败阈值

    CircuitBreakerConfig():m_timeOutSec4Open(10),
                           m_clearStatisticInterval(60),
                           m_minReqCount(20),
                           m_failThreshhold2OpenState(0.4),
                           m_failThreshhold2CloseState(0.1)
    {}

};

class CircuitBreaker // 熔断器的实现
{
private:
    CIRCUIT_BREAKER_STATE m_state; // 熔断器所处的状态
    uint32_t m_failureCnt; // 外部请求的失败和成功计数
    uint32_t m_successCnt;
    CircuitBreakerConfig m_config; //配置项
    mutex m_mutex; 
    thread *m_clearStatisticThread; //负责清理和状态转移的线程
    uint32_t m_threadHeartBeatTime; // 线程的最后heat beat时间，用来检查上面的线程是否还活着
    bool m_stopClearStatisticFlag; //上述线程的停止标志
    default_random_engine m_rnd; //随机数发生器
    uniform_real_distribution<> m_distrib;

    void checkHb()
    {
        if (time(NULL) - m_threadHeartBeatTime > 3)
        {
            throw "clearStatisticThread abnormal!";
        }
    }


public:
    CircuitBreaker(const CircuitBreakerConfig & config)
    {
        new (this)CircuitBreaker();
        m_config = config;
    }
    CircuitBreaker():m_state(CLOSE), 
                    m_failureCnt(0), 
                    m_successCnt(0), 
                    m_stopClearStatisticFlag(false),
                    m_distrib(0, 1.0),
                    m_threadHeartBeatTime(time(NULL)),
                    m_config()

    {
        m_clearStatisticThread = new thread(&CircuitBreaker::clearStatistic, this);
        m_rnd.seed(time(NULL));
    }
    void clearStatistic()
    {
        time_t lastClearTime = time(NULL);
        time_t enterOpenStateTime; // 进入open状态的时刻
        time_t enterHalfOpenStateTime; // 进入半open状态的时刻

        while (!m_stopClearStatisticFlag)
        {
            usleep(5000);
            std::unique_lock<std::mutex> lk(m_mutex);
            time_t current = time(NULL);
            m_threadHeartBeatTime = current;
                
            if (m_state == CLOSE) // 如果是闭合状态，那么检查失败率是否达到一定的阈值，达到了就进入open状态
            {
                if ( (m_failureCnt + m_successCnt) > m_config.m_minReqCount &&
                    (m_failureCnt*1.0 / (m_failureCnt + m_successCnt)) > m_config.m_failThreshhold2OpenState)
                {
                    printf("close->open,  failure rate:%.2f\n", (m_failureCnt*1.0 / (m_failureCnt + m_successCnt)));
                    m_state = OPEN;    
                    enterOpenStateTime = current;
                    // 重置计数器
                    m_failureCnt = 0;
                    m_successCnt = 0;
                    
                }
            }
            else if (m_state == OPEN) // 如果是open状态，一段时间后，进入半open状态
            {
                //printf("in open state, %d\n", current - enterOpenStateTime);
                if (current - enterOpenStateTime > m_config.m_timeOutSec4Open)
                {
                    printf("open->half open, %u, %u, %d\n", m_failureCnt, m_successCnt, current - enterOpenStateTime);
                    m_state = HALF_OPEN;
                    enterHalfOpenStateTime = current;
                    // 重置计数器
                    m_failureCnt = 0;
                    m_successCnt = 0;
                }
            }
            else if (m_state == HALF_OPEN) // 如果是半open状态，如果成功率达到一定阈值，或者超过一定的时间，就进入close状态
            {
                if ((m_failureCnt + m_successCnt) > m_config.m_minReqCount && 
                    (m_failureCnt*1.0 / (m_failureCnt + m_successCnt)) < m_config.m_failThreshhold2CloseState)
                {
                    printf("half open-> close, failure rate:%.2f\n",(m_failureCnt*1.0 / (m_failureCnt + m_successCnt)));
                    m_state = CLOSE;
                    // 重置计数器
                    m_failureCnt = 0;
                    m_successCnt = 0;
                }
                else if ((m_failureCnt + m_successCnt) > m_config.m_minReqCount && (m_failureCnt*1.0 / (m_failureCnt + m_successCnt)) > m_config.m_failThreshhold2OpenState)
                {
                    printf("half open->open,  failure rate:%.2f\n", (m_failureCnt*1.0 / (m_failureCnt + m_successCnt)));
                    m_state = OPEN;    
                    enterOpenStateTime = current;
                    // 重置计数器
                    m_failureCnt = 0;
                    m_successCnt = 0;
                }
                else if (current - enterHalfOpenStateTime > m_config.m_timeOutSec4Open)
                {
                    printf("timeout, half open-> close, %u,%u, %d\n", m_failureCnt, m_successCnt, current - enterHalfOpenStateTime);
                    m_state = CLOSE;
                    // 重置计数器
                    m_failureCnt = 0;
                    m_successCnt = 0;
                }
            }

            if (current - lastClearTime > m_config.m_clearStatisticInterval) // 每隔一段时间后就清理一下计数器，保证计数器反映的是最近的统计信息
            {
                printf("reset statistic: %u, %u, %d\n", m_failureCnt, m_successCnt, current - lastClearTime);
                lastClearTime = current;
                // 重置计数器
                m_failureCnt = 0;
                m_successCnt = 0;
                
            }
           
        }
    }
    ~CircuitBreaker()
    {
        m_stopClearStatisticFlag = true;
        m_clearStatisticThread->join();
        delete m_clearStatisticThread;
        printf("breaker deleted\n");
    }
    void success(uint32_t inc = 1)
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_successCnt += inc;
        checkHb();
    }
    void failure(uint32_t inc = 1)
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_failureCnt += inc;
        checkHb();
    }
    CIRCUIT_BREAKER_STATE getCurrentState()
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        checkHb();
        return m_state;
    } 
    bool canAccessDownStream() // 是否要请求下游
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        checkHb();
        if (m_state == CLOSE)
        {
            return true;
        }
        else if (m_state == OPEN)
        {
            return false;
        }
        else if (m_state == HALF_OPEN)
        {
            return (m_distrib(m_rnd)) > 0.2;
        }
        return true;

    }
    CircuitBreaker(CircuitBreaker &) = delete;
    CircuitBreaker(const CircuitBreaker &) = delete;
    CircuitBreaker & operator=(CircuitBreaker &) = delete;
    CircuitBreaker & operator=(const CircuitBreaker &) = delete;

};

class CircuitBreakerControl // 多个熔断器的包装
{
private:
    std::map<string, CircuitBreaker*> m_breakerList;
    mutex m_mutex; 

public:
    CircuitBreakerControl()
    {

    }
    void addNewCircuitBreaker(const string& breakerName )
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        std::map<string, CircuitBreaker*>::iterator it = m_breakerList.find(breakerName);
        if (it == m_breakerList.end())
        {
            CircuitBreaker * b = new CircuitBreaker();
            if (b == NULL)
            {
                return;
            }
            m_breakerList.insert(std::pair<string, CircuitBreaker*>(breakerName, b));
        }
    }
    CircuitBreaker* getCircuitBreaker(const string& breakerName)
    {
        addNewCircuitBreaker(breakerName);
        std::unique_lock<std::mutex> lk(m_mutex);
        std::map<string, CircuitBreaker*>::iterator it = m_breakerList.find(breakerName);
        if (it != m_breakerList.end())
        {
            return it->second;
        }
        else
        {
            return NULL;
        }
    }
    ~CircuitBreakerControl()
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        std::map<string, CircuitBreaker*>::iterator it;
        for (it = m_breakerList.begin(); it != m_breakerList.end(); ++it)
        {
            delete it->second;
        }
    }

};


#endif
