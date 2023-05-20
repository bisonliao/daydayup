/*
����һ�������ظ����ֵ����� nums �������� ���п��ܵ�ȫ���� ��
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
    set<int> possible_values; //����ڵ���ܵ�ȡֵ
    node_t * parrent;
    set<int>::const_iterator it; //����ڵ����ȡֵ�ı���������
    list<int> path; //�Ӹ�һ·�ߵ�����ڵ㣬  һ·��ȡֵ�б���Ҷ�ӽڵ�Ϳ��Դ�ӡ��һ������
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
    
    // ������ȱ����������ÿ���ڵ��¼�Լ��Ŀ��ܵ�ȡֵ���Ӹ��ڵ㵽�Լ�һ·����ȡֵ
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
        cursor->child->possible_values.erase(value);//�������ڵ�Ŀ���ȡֵ�������ų�����ǰ��һ��ȡֵ
        cursor->child->path = cursor->path;
        cursor->child->path.push_back(value); //�������ڵ��path���ӽڵ㣬��׷�Ӹ��ڵ㵱ǰ��һ��ȡֵ��Ҳ��������̽����path
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
