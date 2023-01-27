/*
����24��

˼·������
1.�ж�4�������Ƿ��ܵõ�24�ǱȽϸ��ӵģ�������������ͨ�����������Ƿ��ܵõ�24���൱�����ˣ���˽��������Ĺؼ�����������4�����ֱ��3�����֣��ٱ���������֡�
2.4->3:��4����������ȡ����(6�ֿ���)�����������㣬�õ����ֵ�ֱ���ʣ�µ�����������ϣ��õ���������
3.3->2:��3����������ȡ������3�ֿ��ܣ������������㣬�õ������ֵ�ֱ���ʣ�µ�һ��������ϣ��õ���������
4.2->��������������ֽ����������㣬�����Եõ�24������true��
*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <iterator>
#include <algorithm>
#include <deque>




using namespace std;

#define SUC_RESULT (1000000)

long getUniqID()// id����׷�ټ���·��
{
    static bool seeded = false;
    if (!seeded)
    {
        srandom(time(NULL));
        seeded = true;
    }
    return random();
}


class operand_t //������
{
    public:
        int val;
        bool isOriginal;// true��ʾ���ʼ��������ݣ�false��ʾ�Ǽ���������м���
        long id;   // id����׷�ټ���·��     
        //operand_t(int v) {val=v; isOriginal = false;}
        //operand_t() {val=0; isOriginal = false;}
};

operand_t operator+(const operand_t & a, const operand_t&b)
{
    operand_t r;
    r.val = a.val + b.val;
    r.isOriginal = false;
    return r;
}

operand_t operator-(const operand_t & a, const operand_t & b)
{
    operand_t r;
    r.val = a.val - b.val;
    r.isOriginal = false;
    return r;
}

operand_t operator*(const operand_t & a, const operand_t & b)
{
    operand_t r;
    r.val = a.val * b.val;
    r.isOriginal = false;
    return r;
}
operand_t operator/(const operand_t & a, const operand_t & b)
{
    operand_t r;
    if (b.val != 0) r.val = a.val / b.val; else r.val = 0;
    r.isOriginal = false;
    return r;
}
operand_t operator%(const operand_t & a, const operand_t & b)
{
    operand_t r;
    if (b.val != 0) r.val = a.val % b.val; else r.val = 0;
    r.isOriginal = false;
    return r;
}

class step_t//һ�μ��㲽��
{
    public:
        operand_t left;
        operand_t right;
        int op;// 1 �� 2�� 3�� 4��
        operand_t res;
        
};
bool operator<(const step_t &a, const step_t &b)//set��Ҫ�ĵײ㺯��
{
    
    const void * p1 = &a;
    const void * p2 = &b;
    return memcmp(p1, p2, sizeof(step_t)) < 0;
    

}
bool operator==(const step_t &a, const step_t &b)
{
    printf("gooooooooooooooood\n");//�������û�б����ã�����Ϊ����set��������ж���
    const void * p1 = &a;
    const void * p2 = &b;
    return memcmp(p1, p2, sizeof(step_t)) == 0;
}

map<int, set<step_t>>  g_stepsSave;//���������̣����ڵ���ʱ��ѯ��

void cleanStepsSave()
{
    g_stepsSave.clear();
}
void insertStepSave(int res, step_t step)//����һ�����㲽��
{
    if (res != step.res.val) {printf("!!!!\n"); exit(-1);}
    map<int, set<step_t>>::iterator it = g_stepsSave.find(res);
    if (it == g_stepsSave.end())
    {
        set<step_t> steps;
        steps.insert(step);
        g_stepsSave.insert(pair<int, set<step_t>>(res, steps));
    }
    else
    {
        it->second.insert(step);
    }
}
int findSolutions(vector<operand_t> & numbers);

void reduce(const vector<operand_t>&numbers, const operand_t* a, const operand_t* b, int j, int k) // 4����3����3����2���Ĺ���
{
    vector<operand_t> leftNum;
    step_t step;
    operand_t result;
    result.isOriginal = false;

    leftNum.clear();
    if (a != NULL)
        leftNum.push_back(*a);
    if (b != NULL)
        leftNum.push_back(*b);
    result = numbers[j] + numbers[k];
    result.id = getUniqID();
    result.isOriginal = false;
    leftNum.push_back(result);// �����������㣬���ں��������һ��������
    step.left = numbers[j];
    step.right = numbers[k];
    step.op = 1;
    step.res = result; // ��¼����м�ֵ����Դ���衣 ͨ��id��step��result���������ˣ����ܼ򵥱Ƚ�res.val == result.val��Ҫ�Ƚ�id��ȷ������·��
    insertStepSave(result.val, step);
    findSolutions(leftNum);

    leftNum.clear();
    if (a != NULL)
        leftNum.push_back(*a);
    if (b != NULL)
        leftNum.push_back(*b);
    result = numbers[j] - numbers[k];
    result.id = getUniqID();
    result.isOriginal = false;
    leftNum.push_back(result);
    step.left = numbers[j];
    step.right = numbers[k];
    step.op = 2;
    step.res = result;
    insertStepSave(result.val, step);
    findSolutions(leftNum);

    leftNum.clear();
    if (a != NULL)
        leftNum.push_back(*a);
    if (b != NULL)
        leftNum.push_back(*b);
    result = numbers[k] - numbers[j];
    result.id = getUniqID();
    result.isOriginal = false;
    leftNum.push_back(result);
    step.left = numbers[k];
    step.right = numbers[j];
    step.op = 2;
    step.res = result;
    insertStepSave(result.val, step);
    findSolutions(leftNum);

    leftNum.clear();
    if (a != NULL)
        leftNum.push_back(*a);
    if (b != NULL)
        leftNum.push_back(*b);
    result = numbers[j] * numbers[k];
    result.id = getUniqID();
    result.isOriginal = false;
    leftNum.push_back(result);
    step.left = numbers[j];
    step.right = numbers[k];
    step.op = 3;
    step.res = result;
    insertStepSave(result.val, step);
    findSolutions(leftNum);

    if (numbers[k].val != 0 && (numbers[j].val % numbers[k].val) == 0)
    {
        leftNum.clear();
        if (a != NULL)
            leftNum.push_back(*a);
        if (b != NULL)
            leftNum.push_back(*b);
        result = numbers[j] / numbers[k];
        result.id = getUniqID();
        result.isOriginal = false;
        leftNum.push_back(result);
        step.left = numbers[j];
        step.right = numbers[k];
        step.op = 4;
        step.res = result;
        insertStepSave(result.val, step);
        findSolutions(leftNum);
    }
    if (numbers[j].val != 0 && (numbers[k].val % numbers[j].val) == 0)
    {
        leftNum.clear();
        if (a != NULL)
            leftNum.push_back(*a);
        if (b != NULL)
            leftNum.push_back(*b);
        result = numbers[k] / numbers[j];
        result.id = getUniqID();
        result.isOriginal = false;
        leftNum.push_back(result);
        step.left = numbers[k];
        step.right = numbers[j];
        step.op = 4;
        step.res = result;
        insertStepSave(result.val, step);
        findSolutions(leftNum);
    }
   
}

int findSolutions(vector<operand_t> & numbers)
{
    #if 1
    operand_t suc;
    suc.val = SUC_RESULT;
    suc.isOriginal =false;
    if (numbers.size() == 2)
    {
        if ( (numbers[0] + numbers[1]).val == 24)
        {
            
            step_t step;
            step.left = numbers[0];
            step.right = numbers[1];
            step.op = 1;
            suc.id = getUniqID();
            step.res = suc;
            insertStepSave(SUC_RESULT, step);
            return 0;
        }
        if ((numbers[0] - numbers[1]).val == 24)
        {
            step_t step;
            step.left = numbers[0];
            step.right = numbers[1];
            step.op = 2;
            suc.id = getUniqID();
            step.res = suc;
            insertStepSave(SUC_RESULT, step);
            return 0;
        }
        if ((numbers[1] - numbers[0]).val == 24)
        {
            step_t step;
            step.left = numbers[1];
            step.right = numbers[0];
            step.op = 2;
            suc.id = getUniqID();
            step.res = suc;
            insertStepSave(SUC_RESULT, step);
            return 0;
        }
        if ( (numbers[0] * numbers[1]).val == 24)
        {
            step_t step;
            step.left = numbers[0];
            step.right = numbers[1];
            step.op = 3;
            suc.id = getUniqID();
            step.res = suc;
            insertStepSave(SUC_RESULT, step);
            return 0;
        }
        if (numbers[1].val != 0 && ( (numbers[0] % numbers[1]).val == 0) && (numbers[0] / numbers[1]).val == 24)
        {
            step_t step;
            step.left = numbers[0];
            step.right = numbers[1];
            step.op = 4;
            suc.id = getUniqID();
            step.res = suc;
            insertStepSave(SUC_RESULT, step);
            return 0;
        }
        if (numbers[0].val != 0 && ( (numbers[1] % numbers[0]).val == 0) && (numbers[1] / numbers[0]).val == 24)
        {
            step_t step;
            step.left = numbers[1];
            step.right = numbers[0];
            step.op = 4;
            suc.id = getUniqID();
            step.res = suc;
            insertStepSave(SUC_RESULT, step);
            return 0;
        }
        return -1;
    }
    else if (numbers.size() == 3)
    {
        for (int i = 0; i < 3; ++i)
        {
            int j = (i+1)%3;
            int k = (i+2)%3;

            reduce(numbers, &numbers[i], NULL, j, k);

        }
    }
    else if (numbers.size() == 4)
    {
        int i,m;
        for (i = 0; i < 4; ++i)
        {
            for (m = i+1; m < 4; ++m)// ���д��for (m=0;m<4;++m)�����ظ�
            {
                if (i == m)
                {
                    continue;
                }
                int j, k;
                set<int> index = {0, 1, 2, 3};
                index.erase(i);
                index.erase(m);
                set<int>::iterator it = index.begin();
                j = *it;
                it++;
                k = *it;

                reduce(numbers, &numbers[i], &numbers[m], j, k);
            }
        }

    }
    else
    {
        printf("invalid input numbers");
        return -1;
    }
    #endif
    return 0;

}

string op2str(int op)
{
    switch (op)
    {
        case 1: return "+";
        case 2: return "-";
        case 3: return "x";
        case 4: return "/";
        default: return "?";
    }
}
string original(const operand_t & o)
{
    if (o.isOriginal) return ".";
    return "";
}

int printSolutions()
{
    map<int, set<step_t>>::iterator it;
    deque<operand_t> resultAtMiddle;
    operand_t suc;
    suc.id =0;
    suc.isOriginal = false;
    suc.val =SUC_RESULT;

    resultAtMiddle.push_front(suc);   
    

    while (resultAtMiddle.size() > 0)
    {
        operand_t res = resultAtMiddle.at(0);
        resultAtMiddle.pop_front();

        it = g_stepsSave.find(res.val);
        if (it == g_stepsSave.end())
        {
            printf("can NOT find solution\n");
            return -1;
        }
        const set<step_t> &steps = it->second;
        set<step_t>::const_iterator it2;
        for (it2 = steps.begin(); it2 != steps.end(); it2++)
        {
            step_t step = *it2;
            if (step.res.val != res.val)
            {
                printf("invalid!\n");
                //exit(-1);
            }
            if (res.val!= SUC_RESULT && res.id != step.res.id)
            {
                continue;
            }
            printf("%d[%ld]=%d[%ld]%s %s %d[%ld]%s\n", res.val, res.id,
                    step.left.val, step.left.id, original(step.left).c_str(),
                   op2str(step.op).c_str(),
                   step.right.val, step.right.id, original(step.right).c_str());
            it++;
            bool bottom = true;
            if (!step.left.isOriginal)
            {
                resultAtMiddle.push_front(step.left);
                bottom = false;
            }
            if (!step.right.isOriginal)
            {
                resultAtMiddle.push_front(step.right);
                bottom = false;
            }
            if (bottom) { printf("----------------\n");}
        }
    }
    return 0;
}


int main()
{
    vector<operand_t> numbers;
    operand_t a;

    numbers.clear();
    a.id = getUniqID(); a.val = 7; a.isOriginal = true;numbers.push_back(a);
    a.id = getUniqID(); a.val = 1; a.isOriginal = true;numbers.push_back(a);
    a.id = getUniqID(); a.val = 8; a.isOriginal = true;numbers.push_back(a);
    a.id = getUniqID(); a.val = 3; a.isOriginal = true;numbers.push_back(a);
    cleanStepsSave();
    findSolutions(numbers);
    printSolutions();

    printf("\n\n");
    numbers.clear();
    a.id = getUniqID(); a.val = 3; a.isOriginal = true;numbers.push_back(a);
    a.id = getUniqID(); a.val = 3; a.isOriginal = true;numbers.push_back(a);
    a.id = getUniqID(); a.val = 3; a.isOriginal = true;numbers.push_back(a);
    a.id = getUniqID(); a.val = 3; a.isOriginal = true;numbers.push_back(a);
    cleanStepsSave();
    findSolutions(numbers);
    printSolutions();

    return 0;
}

