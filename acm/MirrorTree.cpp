/*
*  �ж�һ���������Ƿ���/���ҶԳơ�
*  ��������ͬ�� �����������������Ӧ�ڵ㣬�����ȣ��������ҶԳƵģ�������
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
*  �ж�һ���������Ƿ���/���ҶԳơ�
*  ��������ͬ�� �����������������Ӧ�ڵ㣬�����ȣ��������ҶԳƵģ�������
*/
int isMirrorTree(const Node * root, bool & yesOrNo)
{
    if (root == NULL)
    {
        yesOrNo = true;
        return 0;
    }
    deque<const Node*> nodes1, nodes2;
    if (root->leftChild && root->rightChild) // ��Ҫ�ȶԵ����������ŵ�deque����ϱȶ�
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
        // �ȶ�����deque��ͬ��λ�õ������ڵ㣬�������ȾͲ��ǶԳƵġ��ȶ��꣬�ٰ����ǵ�������Ӧ��ѹ��deque
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


///////////////����ȫ����chatGPT4.0�������ɵĵ�Ԫ��������

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
    Node node6 = {5, nullptr, nullptr};  // �����ֵ��4��Ϊ5���������Ͳ��ٶԳ�
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
    // ���������ǲ����� node6��ʹ��һ����������Ӧλ��Ϊ��
    Node node7 = {3, nullptr, nullptr};
    node1.leftChild = &node2;
    node1.rightChild = &node3;
    node2.leftChild = &node4;
    node2.rightChild = &node5;
    // ���ǲ��ٸ� node3 �����Ӹ�ֵ
    node3.rightChild = &node7;
    bool yesOrNo;
    EXPECT_EQ(0, isMirrorTree(&node1, yesOrNo));
    EXPECT_FALSE(yesOrNo);
}

