/*
����һ���������ĸ��ڵ�?root������ڵ�λ�ڶ������ĵ� 1 �㣬�����ڵ���ӽڵ�λ�ڵ� 2 �㣬�������ơ�
�뷵�ز���Ԫ��֮�� ��� ���Ǽ��㣨����ֻ��һ�㣩�Ĳ��. 

�������ı�ʾΪ��ȫ�����������磺

���룺root = [1,7,0,7,-8,null,null]
�����2
���ͣ�
�� 1 ���Ԫ��֮��Ϊ 1��
�� 2 ���Ԫ��֮��Ϊ 7 + 0 = 7��
�� 3 ���Ԫ��֮��Ϊ 7 + -8 = -1��
�������Ƿ��ص� 2 ��Ĳ�ţ����Ĳ���Ԫ��֮�����
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

#include <vector>
#include <stack>
#include <string>
#include <deque>
#include <map>

using namespace std;



struct node_t
{
    string value;
    struct node_t * left;
    struct node_t * right;
};



node_t * buildTree(const vector<string> & elements)
{
    node_t * root = NULL;
    deque<node_t**> nodes;
    nodes.push_back(&root);

    vector<string>::const_iterator it = elements.begin();
    while (it != elements.end() && nodes.size() > 0)
    {
        node_t ** p = nodes[0];
        nodes.pop_front();

        *p = new node_t();
        (*p)->value = *it;
        (*p)->left = NULL;
        (*p)->right = NULL;

        nodes.push_back(&((*p)->left));
        nodes.push_back(&((*p)->right));

        it++;
    }
    return root;
}
int printTree(const node_t * root)
{
    printf("[");
    deque<const node_t*> nodes;
    nodes.push_back(root);
    while (nodes.size() > 0)
    {
        const node_t * p = nodes[0];
        nodes.pop_front();

        printf("%s,", p->value.c_str());
        if (p->left)  nodes.push_back(p->left);
        if (p->right) nodes.push_back(p->right);

    }
    printf("]\n");
    return 0;
}
#define MAX_LAYER_NUM (1000)

int findMaxLayer(const node_t * root)
{
    map<const node_t *, int> p2layer;
    deque<const node_t*> nodes;
    int sum[MAX_LAYER_NUM] = {0};
    int layer = 1;

    nodes.push_back(root);
    p2layer.insert(pair<const node_t *, int>(root, 1));
    while (nodes.size() > 0)
    {
        const node_t * p = nodes[0];
        nodes.pop_front();
        
        layer = p2layer.find(p)->second;

        if (p->value != "null")
        {
            sum[layer] += atoi(p->value.c_str());
        }

        if (p->left)  
        {
            nodes.push_back(p->left); 
            p2layer.insert(pair<const node_t *, int>(p->left, layer+1));
        }
        if (p->right) 
        {
            nodes.push_back(p->right);
            p2layer.insert(pair<const node_t *, int>(p->right, layer+1));
        }

    }
    int maxIndex = 0;
    int i;
    for (i = 2; i< MAX_LAYER_NUM; ++i)
    {
        if (sum[i] > sum[maxIndex])
        {
            maxIndex = i;
        }
    }
    return maxIndex;
}

int main()
{
    vector<string> elements = {"1", "2", "null", "null", "31", "null", "4", "null", "null", "5", "6"};
    node_t * root = buildTree(elements);

    printTree(root);
    printf(">>%d\n", findMaxLayer(root));
    return 0;

}
