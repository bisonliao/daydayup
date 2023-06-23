/*
*  判断一个二叉树是否镜像/左右对称。
*  方法就是同步 逐个遍历左右子树对应节点，如果相等，则是左右对称的，否则不是
*/
#include <stdlib.h>
#include <deque>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <gtest/gtest.h>

using namespace std;

typedef struct
{
    int value;
    Node * leftChild;
    Node * rightChild;
} Node;
/*
*  判断一个二叉树是否镜像/左右对称。
*  方法就是同步 逐个遍历左右子树对应节点，如果相等，则是左右对称的，否则不是
*/
int isMirrorTree(const Node * root, bool & yesOrNo)
{
    if (root == NULL)
    {
        return -1;
    }
    deque<const Node*> nodes1, nodes2;
    if (root->leftChild && root->rightChild) // 把要比对的左右子树放到deque里，不断比对
    {
        nodes1.push_back(root->leftChild);
        nodes2.push_back(root->rightChild);
    }
    else if (root->leftChild == NULL && root->rightChild == NULL)
    {
        yesOrNo = true;
        return 0;
    }    
    else
    {
        yesOrNo = false;
        return 0;
    }
    
    while (nodes1.size() > 0)
    {
        // 比对两个deque里同样位置的两个节点，如果不相等就不是对称的。比对完，再把他们的子树对应的压入deque
        const Node *a = nodes1.at(0);
        const Node *b = nodes2.at(0);
        nodes1.pop_front();
        nodes2.pop_front();
        if (a->value != b->value)
        {
            yesOrNo = false;
            return 0;
        }
        if (a->leftChild && b->rightChild)
        {
            nodes1.push_back(a->leftChild);
            nodes2.push_back(b->rightChild);
        }
        else if (a->leftChild != NULL && b->rightChild == NULL ||
                a->leftChild == NULL && b->rightChild != NULL)
        {
            yesOrNo = false;
            return 0;
        }
        if (a->rightChild && b->leftChild)
        {
            nodes1.push_back(a->rightChild);
            nodes2.push_back(b->leftChild);
        }
        else if (b->leftChild != NULL && a->rightChild == NULL ||
                b->leftChild == NULL && a->rightChild != NULL)
        {
            yesOrNo = false;
            return 0;
        }
    }
    yesOrNo = true;
    return 0;
}


