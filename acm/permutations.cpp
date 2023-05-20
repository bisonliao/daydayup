/*
给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <set>
#include <list>
#include <unistd.h>

using namespace std;

struct node_t
{
    set<int> possible_values; //这个节点可能的取值
    node_t * parrent;
    set<int>::const_iterator it; //这个节点可能取值的遍历迭代器
    list<int> path; //从根一路走到这个节点，  一路的取值列表，到叶子节点就可以打印出一种排列
    node_t * child;
};


int permutations(const int * array, int len)
{
    node_t * root = new node_t();
    root->possible_values = set<int>(array, array + len);
    root->parrent = NULL;
    root->it = root->possible_values.begin();
    root->child = NULL;

    printf("node size:%lu\n", sizeof(*root));
    
    // 深度优先遍历这棵树，每个节点记录自己的可能的取值、从根节点到自己一路来的取值
    node_t * cursor = root;
    while (true)
    {
        if (cursor->it == cursor->possible_values.end())
        {      
            if (cursor->parrent == NULL) // cursor is root
            {
                if (root->child != NULL) { delete root->child; root->child = NULL;}
                delete root;
                root = NULL;
                return 0;
            }
            else
            {
                node_t * tmp = cursor;
                delete tmp;
                cursor = cursor->parrent;
                cursor->child = NULL; 
                continue;
            }
                
        }
        int value = *cursor->it;
        cursor->it++;

        // create child
        if (cursor->child != NULL) { delete cursor->child; cursor->child = NULL;}
        cursor->child = new node_t();
        cursor->child->possible_values = cursor->possible_values;
        cursor->child->possible_values.erase(value);//拷贝父节点的可能取值，并且排除掉当前的一个取值
        cursor->child->path = cursor->path;
        cursor->child->path.push_back(value); //拷贝父节点的path到子节点，并追加父节点当前的一个取值，也就是往下探索的path
        cursor->child->parrent = cursor;
        cursor->child->child = NULL;
        cursor->child->it = cursor->child->possible_values.begin();

        if (cursor->child->possible_values.size() == 0) // reach the deepest position
        {
            
            list<int>::const_iterator it;
            for (it = cursor->child->path.begin(); it!= cursor->child->path.end(); it++)
            {
                printf("%d ", *it);
            }
            printf("\n");
            delete cursor->child;
        }
        else
        {
            cursor = cursor->child; // go down, going deep first
        }
   
    }
    
    return 0;
}


int main()
{
    int a[] = {1, 2, 3, 4,5,6,7,8,9,10};
    int len  = sizeof(a)/sizeof(int);
    permutations(a, len);
   
    return 0;
}
