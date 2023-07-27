/*
* 写一个函数，返回二叉树的深度
*  算法一：类似后续遍历
*  算法二：把它按照左右关系打平放到数组里，检查数组的最后一个元素的下标范围可得出深度
*  算法三：广度优先遍历，每次遍历，深度加1
* g++ -ottx TreeDepth.cpp -I/usr/src/googletest/googlemock/include  -lgtest -lpthread
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stack>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <stack>
#include <queue>
#include <cmath>

using namespace std;


class Node
{
    public:
       int value;
       Node * left;
       Node * right;
       int depth;
       Node() {
            left = NULL;
            right = NULL;
            depth = -1;
        }
       Node(int data) {
           value = data;
           left = NULL;
           right = NULL;
           depth = -1;
       }
       Node(int data, Node * right, Node * left) {
           value = data;
           this->left = left;
           this->right = right;
           depth = -1;
       }
};
// 算法三：广度优先遍历，每次遍历，深度加1
int getTreeDepth3(Node * root)
{
    if (root == NULL) {return 0;}
    deque<Node *> scanList;
    root->depth = 1;
    scanList.push_back(root);
    int depth = -1;
    while (!scanList.empty())
    {
        Node * p = scanList.front();
        scanList.pop_front();
        if (depth < p->depth) {depth = p->depth;}

        if (p->left)
        {
            p->left->depth = p->depth + 1;
            scanList.push_back(p->left);
        }
        if (p->right)
        {
            p->right->depth = p->depth + 1;
            scanList.push_back(p->right);
        }

    }
    return depth;

}
//算法二：把它按照左右关系打平放到数组里，检查数组的最后一个元素的下标范围可得出深度
int getTreeDepth2(const Node * root)
{
    vector<const Node *> arr;
    deque<const Node * > scanList;
    if (root == NULL) { return 0;}
    scanList.push_back(root);
    int validNodeNumInScanList = 1;
    while (!scanList.empty() && validNodeNumInScanList > 0)
    {
        const Node * p = scanList.front();
        scanList.pop_front();
        if (p != NULL)  { validNodeNumInScanList--;}

        arr.push_back(p);
        if (p == NULL)
        {
            scanList.push_back(NULL);
            scanList.push_back(NULL);
        }
        else
        {
            scanList.push_back(p->left); if (p->left) { validNodeNumInScanList++;}
            scanList.push_back(p->right);if (p->right) { validNodeNumInScanList++;}
        }
    }
    if (arr.size() < 1) { return 0;}
    vector<const Node*>::const_iterator it = arr.end();
    it--;
    while (it != arr.begin())
    {
        if (*it != NULL)
        {
            break;
        }
    }
    int index = it - arr.begin() ;
    int depth = 0;
    int sum = 0;
    while (true)
    {
        sum += pow(2, depth);
        if (sum > index)
        {
            return depth+1;
        }
        depth++;
    }
    return -1;
}

// 算法一：类似后序遍历的方式
int getTreeDepth(Node * root)
{
    if (root == NULL) { return 0;}
    Node * p = root;
    stack<Node*> path;
    while (true)
    {
        if (p->left != NULL) 
        {          
            path.push(p);
            p = p->left;
        }
        else if (p->right != NULL)
        {
            path.push(p);
            p = p ->right;
        }
        else // leaf node
        { 
            p->depth = 1;

        go_up:
            if (path.empty())
            {
                if (p == root) { return p->depth;}
                else { return -1;}      
            }
            Node * parent = path.top();
            path.pop();
            if (parent->depth < (p->depth+1))
            {
                parent->depth = p->depth+1;
            }
            if (parent->left == p)
            {
                if (parent->right == NULL)
                {
                    p = parent;
                    goto go_up;
                }
                else
                {
                    path.push(parent);
                    p = parent->right;
                } 
            }
            else if (parent->right == p)
            {
                p = parent;
                goto go_up;
            }


        }
    }
    return -2;
}
int freeTree(const Node * root)
{
    if (root == NULL) { return 0;}
    const Node * p = root;
    stack<const Node*> path;
    while (true)
    {
        if (p->left != NULL) 
        {          
            path.push(p);
            p = p->left;
        }
        else if (p->right != NULL)
        {
            path.push(p);
            p = p ->right;
        }
        else // leaf node
        { 

        go_up:
            if (path.empty())
            {
                if (p == root) { delete p; return 0;}
                else { return -1;}      
            }
            const Node * parent = path.top();
            path.pop();
        
            if (parent->left == p)
            {
                if (parent->right == NULL)
                {
                    delete p;
                    p = parent;
                    goto go_up;
                }
                else
                {
                    path.push(parent);
                    delete p;
                    p = parent->right;
                } 
            }
            else if (parent->right == p)
            {
                delete p;
                p = parent;
                goto go_up;
            }


        }
    }
    return -2;
}

// Test fixture for Tree tests
class TreeTest : public ::testing::Test
{
public:
    // Helper function to create a binary tree with given values
    static Node* createBinaryTree(const std::vector<int>& values, size_t index)
    {
        if (index >= values.size() || values[index] == -1) // -1 represents NULL node
            return nullptr;

        Node* root = new Node(values[index]);
        root->left = createBinaryTree(values, 2 * index + 1);
        root->right = createBinaryTree(values, 2 * index + 2);
        return root;
    }
};

// Test cases for getTreeDepth3 function
TEST(getTreeDepth3, EmptyTree)
{
    // Test case for an empty tree (null root)
    Node* root = nullptr;
    int depth = getTreeDepth3(root);
    EXPECT_EQ(depth, 0);
}

TEST(getTreeDepth3, SingleNodeTree)
{
    // Test case for a tree with only one node
    Node* root = new Node(42);
    int depth = getTreeDepth3(root);
    EXPECT_EQ(depth, 1);
    freeTree(root);
}

TEST(getTreeDepth3, FullBinaryTree)
{
    // Test case for a full binary tree with depth 3
    std::vector<int> values = {1, 2, 3, 4, 5, 6, 7};
    Node* root = TreeTest::createBinaryTree(values, 0);
    int depth = getTreeDepth3(root);
    EXPECT_EQ(depth, 3);
    freeTree(root);
}

TEST(getTreeDepth3, SkewedTree)
{
    // Test case for a skewed binary tree with depth 4
    std::vector<int> values = {1, -1, 2, -1,-1, -1, 3,-1,-1,-1,-1,-1,-1, -1, 4};
    Node* root = TreeTest::createBinaryTree(values, 0);
    int depth = getTreeDepth3(root);
    EXPECT_EQ(depth, 4);
    freeTree(root);
}



// Tests for getTreeDepth2 function
TEST(getTreeDepthTest2, EmptyTree)
{
    Node* root = NULL;
    int depth = getTreeDepth2(root);
    EXPECT_EQ(depth, 0);
    freeTree(root);
}

TEST(getTreeDepthTest2, NonEmptyTree)
{
    // Create a simple binary tree for testing
    Node* root = new Node(1);
    root->left = new Node(2);
    root->right = new Node(3);
    root->left->left = new Node(4);
    root->left->right = new Node(5);
    root->right->right = new Node(6);

    int depth = getTreeDepth2(root);
    EXPECT_EQ(depth, 3);
    freeTree(root);
}

TEST(getTreeDepthTest2, SingleNodeTree)
{
    Node* root = new Node(10);
    int depth = getTreeDepth2(root);
    EXPECT_EQ(depth, 1);
    freeTree(root);
}

TEST(getTreeDepthTest2, LeftSkewedTree)
{
    // Create a left-skewed tree: 1 -> 2 -> 3 -> 4
    Node* root = new Node(1);
    root->left = new Node(2);
    root->left->left = new Node(3);
    root->left->left->left = new Node(4);

    int depth = getTreeDepth2(root);
    EXPECT_EQ(depth, 4);
    freeTree(root);
}

TEST(getTreeDepthTest2, RightSkewedTree)
{
    // Create a right-skewed tree: 1 -> 2 -> 3 -> 4
    Node* root = new Node(1);
    root->right = new Node(2);
    root->right->right = new Node(3);
    root->right->right->right = new Node(4);

    int depth = getTreeDepth2(root);
    EXPECT_EQ(depth, 4);
    freeTree(root);
}


/// chatGPT create unit test for me     
Node* createFullTree()
{
    // Create a simple binary tree for testing
    Node* root = new Node(1);
    root->left = new Node(2);
    root->right = new Node(3);
    root->left->left = new Node(4);
    root->left->right = new Node(5);
    root->right->left = new Node(6);
    root->right->right = new Node(7);
    return root;
}
TEST(getTreeDepthTest, FullTree)
{
    Node* root = createFullTree();
    int depth = getTreeDepth(root);
    EXPECT_EQ(depth, 3);
    freeTree(root);
}                         

TEST(getTreeDepthTest, PositiveNos) { 
    Node* root = new Node(3);
    root->left = new Node(9);
    root->right = new Node(20);
    root->right->left = new Node(15);
    root->right->right = new Node(7);

    EXPECT_EQ(getTreeDepth(root), 3); 
    freeTree(root);
}

TEST(getTreeDepthTest, NegativeNos) { 
    EXPECT_EQ(getTreeDepth(NULL), 0); 
}

TEST(getTreeDepthTest, OnlyRoot) { 
    Node* root = new Node(3);
    EXPECT_EQ(getTreeDepth(root), 1); 
    freeTree(root);
}

TEST(getTreeDepthTest, OnlyLeft) { 
    Node* root = new Node(3);
    root->left = new Node(9);

    EXPECT_EQ(getTreeDepth(root), 2); 
    freeTree(root);
}

TEST(getTreeDepthTest, OnlyRight) { 
    Node* root = new Node(3);
    root->right = new Node(20);

    EXPECT_EQ(getTreeDepth(root), 2); 
    freeTree(root);
}

TEST(getTreeDepthTest, LeftSkewedTree)
{
    // Create a left-skewed tree: 1 -> 2 -> 3 -> 4
    Node* root = new Node(1);
    root->left = new Node(2);
    root->left->left = new Node(3);
    root->left->left->left = new Node(4);

    int depth = getTreeDepth(root);
    EXPECT_EQ(depth, 4);
    freeTree(root);
}

TEST(getTreeDepthTest, RightSkewedTree)
{
    // Create a right-skewed tree: 1 -> 2 -> 3 -> 4
    Node* root = new Node(1);
    root->right = new Node(2);
    root->right->right = new Node(3);
    root->right->right->right = new Node(4);

    int depth = getTreeDepth(root);
    EXPECT_EQ(depth, 4);
    freeTree(root);
}


TEST(getTreeDepthTest, Unbalanced) { 
    Node* root = new Node(3);
    root->left = new Node(9);
    root->left->left = new Node(10);

    EXPECT_EQ(getTreeDepth(root), 3); 
    freeTree(root);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
