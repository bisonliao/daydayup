/*
 * 给你一个整数数组?nums，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。你可以按 任意顺序 返回答案。
 * 你必须设计并实现线性时间复杂度的算法且仅使用常量额外空间来解决此问题。
 * 思路：
 * 先异或一轮，得到结果，结果的为1的位，说明要找的这两个元素在这个位取值一个为1 一个为0
 * 按照这个位为1还是为0，把数组元素分成两组，要找的两个元素就分别在这两个小组里，只要把他们分开了，就可以再用异或把他们找出来了。
 * 其他出现两次的元素，都是确定性的出现在其中一个组，他们在小组内的异或结果会是0
 *
 * 很有意思的是，我用蹩脚的英语写明函数的功用和函数的原型的时候，starcoder就能逐步生成这个代码，他知道这个算法！
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

