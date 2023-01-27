/*
计算24点

思路：倒推
1.判断4个数字是否能得到24是比较复杂的，但是两个数字通过四则运算是否能得到24就相当容易了，因此解决此问题的关键在于怎样把4个数字变成3个数字，再变成两个数字。
2.4->3:从4个数字中任取两个(6种可能)进行四则运算，得到五个值分别与剩下的两个数字组合，得到三个数；
3.3->2:从3个数字中任取两个（3种可能）进行四则运算，得到的五个值分别与剩下的一个数字组合，得到两个数；
4.2->结果：将两个数字进行四则运算，若可以得到24，返回true。
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

long getUniqID()// id用于追踪计算路径
{
    static bool seeded = false;
    if (!seeded)
    {
        srandom(time(NULL));
        seeded = true;
    }
    return random();
}


class operand_t //操作数
{
    public:
        int val;
        bool isOriginal;// true表示是最开始输入的数据，false表示是计算得来的中间结果
        long id;   // id用于追踪计算路径     
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

class step_t//一次计算步骤
{
    public:
        operand_t left;
        operand_t right;
        int op;// 1 加 2减 3乘 4除
        operand_t res;
        
};
bool operator<(const step_t &a, const step_t &b)//set需要的底层函数
{
    
    const void * p1 = &a;
    const void * p2 = &b;
    return memcmp(p1, p2, sizeof(step_t)) < 0;
    

}
bool operator==(const step_t &a, const step_t &b)
{
    printf("gooooooooooooooood\n");//这个函数没有被调用，我以为放在set里会用来判断呢
    const void * p1 = &a;
    const void * p2 = &b;
    return memcmp(p1, p2, sizeof(step_t)) == 0;
}

map<int, set<step_t>>  g_stepsSave;//缓存计算过程，用于倒推时查询用

void cleanStepsSave()
{
    g_stepsSave.clear();
}
void insertStepSave(int res, step_t step)//插入一个计算步骤
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

void reduce(const vector<operand_t>&numbers, const operand_t* a, const operand_t* b, int j, int k) // 4个变3个，3个变2个的过程
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
    leftNum.push_back(result);// 继续后续计算，用于后续计算的一个操作数
    step.left = numbers[j];
    step.right = numbers[k];
    step.op = 1;
    step.res = result; // 记录这个中间值的来源步骤。 通过id把step和result关联起来了，不能简单比较res.val == result.val，要比较id才确定计算路径
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
            for (m = i+1; m < 4; ++m)// 如果写作for (m=0;m<4;++m)会有重复
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

