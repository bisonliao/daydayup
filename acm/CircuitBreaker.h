// �۶�����ʵ�֣�������ܿɼ���https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker
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

class CircuitBreakerConfig //�۶���������
{
public:
    uint32_t m_timeOutSec4Open; // open/half open״̬�µĳ�ʱʱ�� s
    uint32_t m_clearStatisticInterval; //�����������ʱ���� s
    uint32_t m_minReqCount; // ͳ����Ϣ��Ч��Ҫ�����ٵ�����������������̫�پͲ�����Ч
    float m_failThreshhold2OpenState; // תopen״̬��ʧ����ֵ
    float m_failThreshhold2CloseState; // תclose״̬��ʧ����ֵ

    CircuitBreakerConfig():m_timeOutSec4Open(10),
                           m_clearStatisticInterval(60),
                           m_minReqCount(20),
                           m_failThreshhold2OpenState(0.4),
                           m_failThreshhold2CloseState(0.1)
    {}

};

class CircuitBreaker // �۶�����ʵ��
{
private:
    CIRCUIT_BREAKER_STATE m_state; // �۶���������״̬
    uint32_t m_failureCnt; // �ⲿ�����ʧ�ܺͳɹ�����
    uint32_t m_successCnt;
    CircuitBreakerConfig m_config; //������
    mutex m_mutex; 
    thread *m_clearStatisticThread; //���������״̬ת�Ƶ��߳�
    uint32_t m_threadHeartBeatTime; // �̵߳����heat beatʱ�䣬�������������߳��Ƿ񻹻���
    bool m_stopClearStatisticFlag; //�����̵߳�ֹͣ��־
    default_random_engine m_rnd; //�����������
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
        time_t enterOpenStateTime; // ����open״̬��ʱ��
        time_t enterHalfOpenStateTime; // �����open״̬��ʱ��

        while (!m_stopClearStatisticFlag)
        {
            usleep(5000);
            std::unique_lock<std::mutex> lk(m_mutex);
            time_t current = time(NULL);
            m_threadHeartBeatTime = current;
                
            if (m_state == CLOSE) // ����Ǳպ�״̬����ô���ʧ�����Ƿ�ﵽһ������ֵ���ﵽ�˾ͽ���open״̬
            {
                if ( (m_failureCnt + m_successCnt) > m_config.m_minReqCount &&
                    (m_failureCnt*1.0 / (m_failureCnt + m_successCnt)) > m_config.m_failThreshhold2OpenState)
                {
                    printf("close->open,  failure rate:%.2f\n", (m_failureCnt*1.0 / (m_failureCnt + m_successCnt)));
                    m_state = OPEN;    
                    enterOpenStateTime = current;
                    // ���ü�����
                    m_failureCnt = 0;
                    m_successCnt = 0;
                    
                }
            }
            else if (m_state == OPEN) // �����open״̬��һ��ʱ��󣬽����open״̬
            {
                //printf("in open state, %d\n", current - enterOpenStateTime);
                if (current - enterOpenStateTime > m_config.m_timeOutSec4Open)
                {
                    printf("open->half open, %u, %u, %d\n", m_failureCnt, m_successCnt, current - enterOpenStateTime);
                    m_state = HALF_OPEN;
                    enterHalfOpenStateTime = current;
                    // ���ü�����
                    m_failureCnt = 0;
                    m_successCnt = 0;
                }
            }
            else if (m_state == HALF_OPEN) // ����ǰ�open״̬������ɹ��ʴﵽһ����ֵ�����߳���һ����ʱ�䣬�ͽ���close״̬
            {
                if ((m_failureCnt + m_successCnt) > m_config.m_minReqCount && 
                    (m_failureCnt*1.0 / (m_failureCnt + m_successCnt)) < m_config.m_failThreshhold2CloseState)
                {
                    printf("half open-> close, failure rate:%.2f\n",(m_failureCnt*1.0 / (m_failureCnt + m_successCnt)));
                    m_state = CLOSE;
                    // ���ü�����
                    m_failureCnt = 0;
                    m_successCnt = 0;
                }
                else if ((m_failureCnt + m_successCnt) > m_config.m_minReqCount && (m_failureCnt*1.0 / (m_failureCnt + m_successCnt)) > m_config.m_failThreshhold2OpenState)
                {
                    printf("half open->open,  failure rate:%.2f\n", (m_failureCnt*1.0 / (m_failureCnt + m_successCnt)));
                    m_state = OPEN;    
                    enterOpenStateTime = current;
                    // ���ü�����
                    m_failureCnt = 0;
                    m_successCnt = 0;
                }
                else if (current - enterHalfOpenStateTime > m_config.m_timeOutSec4Open)
                {
                    printf("timeout, half open-> close, %u,%u, %d\n", m_failureCnt, m_successCnt, current - enterHalfOpenStateTime);
                    m_state = CLOSE;
                    // ���ü�����
                    m_failureCnt = 0;
                    m_successCnt = 0;
                }
            }

            if (current - lastClearTime > m_config.m_clearStatisticInterval) // ÿ��һ��ʱ��������һ�¼���������֤��������ӳ���������ͳ����Ϣ
            {
                printf("reset statistic: %u, %u, %d\n", m_failureCnt, m_successCnt, current - lastClearTime);
                lastClearTime = current;
                // ���ü�����
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
    bool canAccessDownStream() // �Ƿ�Ҫ��������
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

class CircuitBreakerControl // ����۶����İ�װ
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
