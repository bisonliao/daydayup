/*
 * ����һ����������?nums������ǡ��������Ԫ��ֻ����һ�Σ���������Ԫ�ؾ��������Ρ� �ҳ�ֻ����һ�ε�������Ԫ�ء�����԰� ����˳�� ���ش𰸡�
 * �������Ʋ�ʵ������ʱ�临�Ӷȵ��㷨�ҽ�ʹ�ó�������ռ�����������⡣
 * ˼·��
 * �����һ�֣��õ�����������Ϊ1��λ��˵��Ҫ�ҵ�������Ԫ�������λȡֵһ��Ϊ1 һ��Ϊ0
 * �������λΪ1����Ϊ0��������Ԫ�طֳ����飬Ҫ�ҵ�����Ԫ�ؾͷֱ���������С���ֻҪ�����Ƿֿ��ˣ��Ϳ����������������ҳ����ˡ�
 * �����������ε�Ԫ�أ�����ȷ���Եĳ���������һ���飬������С���ڵ����������0
 *
 * ������˼���ǣ��������ŵ�Ӣ��д�������Ĺ��úͺ�����ԭ�͵�ʱ��starcoder����������������룬��֪������㷨��
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>


/*
* the integer array, all elements in it appear twice except two element
* this function will find the two element that appears only once and set them to a,b
*/
int FindElementAppearsOnlyOnce(const int * array, int len, int *a, int *b)
{
    int xorsum = array[0];
    for (int i = 1; i < len; i++)
    {
        xorsum = xorsum ^ array[i];
    }
    if (xorsum == 0)
    {
        return -1;
    }
    
    // find the lowest bit that is 1 (lowbit)
    int firstbit = 1; // find the lowest bit that is 1 in xorsum
    while ( (firstbit & xorsum) == 0 )
    {
        firstbit = firstbit << 1;
    }
    
    int first = 0, second = 0;
    // the two elements that appears only once will has different value of the lowest bit
    // all elements who has lowest bit 1 will be XORed into 'first'
    // all elements who has lowest bit 0 will be XORed into 'second'
    // the two elements that appears only once will be 'first' and 'second'
    for (int i = 0; i < len; i++)
    {
        if (array[i] & firstbit)
        {
            first = first ^ array[i];
        }
        else
        {
            second = second ^ array[i];
        }
    }
    *a = first;
    *b = second;
    return 0;
}
int main()
{
    //test array
    int array[] = {2, 4, 3, 6, 3, 2, 5, 5};
    int size = sizeof(array) / sizeof(array[0]);
    int a, b;
    if (FindElementAppearsOnlyOnce(array, size, &a, &b) == 0)
    {
        printf("a = %d, b = %d\n", a, b);
    }
    return 0;
}

