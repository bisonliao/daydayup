/*
给你一个二叉树的根节点?root。设根节点位于二叉树的第 1 层，而根节点的子节点位于第 2 层，依此类推。
请返回层内元素之和 最大 的那几层（可能只有一层）的层号. 

二叉树的表示为完全二叉树，例如：

输入：root = [1,7,0,7,-8,null,null]
输出：2
解释：
第 1 层各元素之和为 1，
第 2 层各元素之和为 7 + 0 = 7，
第 3 层各元素之和为 7 + -8 = -1，
所以我们返回第 2 层的层号，它的层内元素之和最大。
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
