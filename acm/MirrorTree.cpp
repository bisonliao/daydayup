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
#include <gmock/gmock.h>

using namespace std;

class Node
{
    public:
    int value;
    Node * leftChild;
    Node * rightChild;
    Node(int v, Node* l, Node* r)
    {
        this->value = v;
        this->leftChild = l;
        this->rightChild = r;
    }
};
/*
*  判断一个二叉树是否镜像/左右对称。
*  方法就是同步 逐个遍历左右子树对应节点，如果相等，则是左右对称的，否则不是
*/
int isMirrorTree(const Node * root, bool & yesOrNo)
{
    if (root == NULL)
    {
        yesOrNo = true;
        return 0;
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


///////////////下面全都是chatGPT4.0帮我生成的单元测试用例

TEST(MirrorTreeTest, NullTree) {
    const Node* root = nullptr;
    bool yesOrNo;
    EXPECT_EQ(0, isMirrorTree(root, yesOrNo));
    EXPECT_TRUE(yesOrNo);
}

TEST(MirrorTreeTest, SingleNodeTree) {
    Node root = {0, nullptr, nullptr};
    bool yesOrNo;
    EXPECT_EQ(0, isMirrorTree(&root, yesOrNo));
    EXPECT_TRUE(yesOrNo);
}

TEST(MirrorTreeTest, TwoNodeTree) {
    Node root = {0, nullptr, nullptr};
    Node left = {0, nullptr, nullptr};
    root.leftChild = &left;
    bool yesOrNo;
    EXPECT_EQ(0, isMirrorTree(&root, yesOrNo));
    EXPECT_FALSE(yesOrNo);
}

TEST(MirrorTreeTest, SymmetricTree) {
    Node root = {0, nullptr, nullptr};
    Node left = {1, nullptr, nullptr};
    Node right = {1, nullptr, nullptr};
    root.leftChild = &left;
    root.rightChild = &right;
    bool yesOrNo;
    EXPECT_EQ(0, isMirrorTree(&root, yesOrNo));
    EXPECT_TRUE(yesOrNo);
}

TEST(MirrorTreeTest, NonSymmetricTree) {
    Node root = {0, nullptr, nullptr};
    Node left = {0, nullptr, nullptr};
    Node right = {1, nullptr, nullptr};
    root.leftChild = &left;
    root.rightChild = &right;
    bool yesOrNo;
    EXPECT_EQ(0, isMirrorTree(&root, yesOrNo));
    EXPECT_FALSE(yesOrNo);
}

TEST(MirrorTreeTest, LargeSymmetricTree) {
    Node node1 = {1, nullptr, nullptr};
    Node node2 = {2, nullptr, nullptr};
    Node node3 = {2, nullptr, nullptr};
    Node node4 = {3, nullptr, nullptr};
    Node node5 = {4, nullptr, nullptr};
    Node node6 = {4, nullptr, nullptr};
    Node node7 = {3, nullptr, nullptr};
    node1.leftChild = &node2;
    node1.rightChild = &node3;
    node2.leftChild = &node4;
    node2.rightChild = &node5;
    node3.leftChild = &node6;
    node3.rightChild = &node7;
    bool yesOrNo;
    EXPECT_EQ(0, isMirrorTree(&node1, yesOrNo));
    EXPECT_TRUE(yesOrNo);
}
TEST(MirrorTreeTest, LargeNonSymmetricTree) {
    Node node1 = {1, nullptr, nullptr};
    Node node2 = {2, nullptr, nullptr};
    Node node3 = {2, nullptr, nullptr};
    Node node4 = {3, nullptr, nullptr};
    Node node5 = {4, nullptr, nullptr};
    Node node6 = {5, nullptr, nullptr};  // 这里的值从4改为5，所以树就不再对称
    Node node7 = {3, nullptr, nullptr};
    node1.leftChild = &node2;
    node1.rightChild = &node3;
    node2.leftChild = &node4;
    node2.rightChild = &node5;
    node3.leftChild = &node6;
    node3.rightChild = &node7;
    bool yesOrNo;
    EXPECT_EQ(0, isMirrorTree(&node1, yesOrNo));
    EXPECT_FALSE(yesOrNo);
}
TEST(MirrorTreeTest, NonSymmetricTreeWithNullNodes) {
    Node node1 = {1, nullptr, nullptr};
    Node node2 = {2, nullptr, nullptr};
    Node node3 = {2, nullptr, nullptr};
    Node node4 = {3, nullptr, nullptr};
    Node node5 = {4, nullptr, nullptr};
    // 在这里我们不定义 node6，使得一侧子树的相应位置为空
    Node node7 = {3, nullptr, nullptr};
    node1.leftChild = &node2;
    node1.rightChild = &node3;
    node2.leftChild = &node4;
    node2.rightChild = &node5;
    // 我们不再给 node3 的左孩子赋值
    node3.rightChild = &node7;
    bool yesOrNo;
    EXPECT_EQ(0, isMirrorTree(&node1, yesOrNo));
    EXPECT_FALSE(yesOrNo);
}

