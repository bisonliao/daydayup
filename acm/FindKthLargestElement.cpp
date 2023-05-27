#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stack>
#include <gtest/gtest.h>

using namespace std;

// 把一个数组进行元素间的位置调整，使得符合最小堆的要求
void makeMinHeap(vector<int>& nums)
{
    if (nums.size() == 0) return;
    stack<int> s;
    s.push(0); 

    while (s.size() > 0)
    {
        int h, l, r;
        h = s.top();
        s.pop();
        l = 2*h + 1;
        r = 2*h + 2;
        if (l < nums.size())
        {
            s.push(l);
        }
        if (r < nums.size())
        {
            s.push(r);
        }

        if (l < nums.size() && nums[l] < nums[h])
        {
            swap(nums[l], nums[h]);
        }
        if (r < nums.size() && nums[r] < nums[h])
        {
            swap(nums[r], nums[h]);
        }
    }
}
// nums本来是符合最小堆的数组，但下标为0的元素也就是根元素被外部代码更新了
// 下面这个函数对nums进行调整，使得继续符合最小堆的要求
void adjustMinHeap(vector<int>&nums)
{
    if (nums.size() == 0) return;
    stack<int> s;
    s.push(0);

    while (s.size() > 0)
    {
        int h, l, r;
        h = s.top();
        s.pop();
        l = 2*h + 1;
        r = 2*h + 2;

        if (l < nums.size() && nums[l] < nums[h])
        {
            swap(nums[l], nums[h]);
            s.push(l);
        }
        if (r < nums.size() && nums[r] < nums[h])
        {
            swap(nums[r], nums[h]);
            s.push(r);
        }
    }
}

// 查找nums这个数组里第K大的最大元素。使用了尺寸受限的最小堆算法
int FindKthLargest(const vector<int>& nums, int k, int & value) 
{
    if (k > nums.size() || nums.size() < 1 || k < 1) return -1;
    vector<int>  minHeapSizeK;
    for(int i = 0; i < k; i++)
    {
        minHeapSizeK.push_back(nums[i]);
    }
    makeMinHeap(minHeapSizeK);
    for(int i = k; i < nums.size(); i++)
    {
        if(nums[i] > minHeapSizeK[0])
        {
            minHeapSizeK[0] = nums[i];
            adjustMinHeap(minHeapSizeK);
        }
    }
    value = minHeapSizeK[0];
    return 0;
}

// 下面都是chaGPT4帮我生成的单元测试，都能通过。

TEST(MinHeapTest, EmptyVector) {
    vector<int> nums;
    makeMinHeap(nums);
    EXPECT_TRUE(nums.empty());
}

TEST(MinHeapTest, SingleElement) {
    vector<int> nums = {5};
    makeMinHeap(nums);
    EXPECT_EQ(nums[0], 5);
}

TEST(MinHeapTest, RandomElements) {
    vector<int> nums = {3, 1, 4, 1, 5, 9, 2, 6, 5};
    makeMinHeap(nums);
    
    // Check if nums satisfies min heap property
    for (int i = 0; i < nums.size(); ++i) {
        int leftChild = 2 * i + 1;
        int rightChild = 2 * i + 2;
        
        if (leftChild < nums.size()) {
            EXPECT_LE(nums[i], nums[leftChild]);
        }
        if (rightChild < nums.size()) {
            EXPECT_LE(nums[i], nums[rightChild]);
        }
    }
}

bool checkMinHeap(const vector<int>& nums) {
    for (int i = 0; i < nums.size(); ++i) {
        int leftChild = 2 * i + 1;
        int rightChild = 2 * i + 2;
        
        if (leftChild < nums.size() && nums[i] > nums[leftChild]) {
            return false;
        }
        if (rightChild < nums.size() && nums[i] > nums[rightChild]) {
            return false;
        }
    }
    return true;
}

TEST(AdjustMinHeapTest, EmptyVector) {
    vector<int> nums;
    adjustMinHeap(nums);
    EXPECT_TRUE(nums.empty());
}

TEST(AdjustMinHeapTest, SingleElement) {
    vector<int> nums = {5};
    adjustMinHeap(nums);
    EXPECT_EQ(nums[0], 5);
}

TEST(AdjustMinHeapTest, HeapPropertyMaintained) {
    vector<int> nums = {10, 20, 30, 40, 50, 60, 70};
    nums[0] = 80; // Disrupt the heap property
    adjustMinHeap(nums);
    EXPECT_TRUE(checkMinHeap(nums));
}


TEST(AdjustMinHeapTest, AscendingOrderInput) {
    vector<int> nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    nums[0] = 0; // Disrupt the heap property
    adjustMinHeap(nums);
    EXPECT_TRUE(checkMinHeap(nums));
}

TEST(AdjustMinHeapTest, LargeNumbersInput) {
    vector<int> nums = {10000, 20000, 30000, 40000, 50000};
    nums[0] = 60000; // Disrupt the heap property
    adjustMinHeap(nums);
    EXPECT_TRUE(checkMinHeap(nums));
}

TEST(AdjustMinHeapTest, RepeatedNumbersInput) {
    vector<int> nums = {5, 5, 5, 5, 5};
    nums[0] = 6; // Disrupt the heap property
    adjustMinHeap(nums);
    EXPECT_TRUE(checkMinHeap(nums));
}


TEST(FindKthLargestTest, EmptyVector) {
    vector<int> nums;
    int value;
    EXPECT_EQ(FindKthLargest(nums, 1, value), -1);
}

TEST(FindKthLargestTest, SingleElement) {
    vector<int> nums = {5};
    int value;
    EXPECT_EQ(FindKthLargest(nums, 1, value), 0);
    EXPECT_EQ(value, 5);
}

TEST(FindKthLargestTest, NormalCase) {
    vector<int> nums = {3,2,1,5,6,4};
    int value;
    EXPECT_EQ(FindKthLargest(nums, 2, value), 0);
    EXPECT_EQ(value, 5);
}

TEST(FindKthLargestTest, LargeInput) {
    vector<int> nums = {3,2,3,1,2,4,5,5,6};
    int value;
    EXPECT_EQ(FindKthLargest(nums, 4, value), 0);
    EXPECT_EQ(value, 4);
}

TEST(FindKthLargestTest, DuplicateElements) {
    vector<int> nums = {3, 2, 3, 1, 2, 4, 5, 5, 6};
    int value;
    EXPECT_EQ(FindKthLargest(nums, 1, value), 0);
    EXPECT_EQ(value, 6);
    EXPECT_EQ(FindKthLargest(nums, 2, value), 0);
    EXPECT_EQ(value, 5);
    EXPECT_EQ(FindKthLargest(nums, 3, value), 0);
    EXPECT_EQ(value, 5);
    EXPECT_EQ(FindKthLargest(nums, 4, value), 0);
    EXPECT_EQ(value, 4);
}

TEST(FindKthLargestTest, KExceedsArraySize) {
    vector<int> nums = {3, 2, 3, 1, 2, 4, 5, 5, 6};
    int value;
    EXPECT_EQ(FindKthLargest(nums, nums.size() + 1, value), -1);
}

TEST(FindKthLargestTest, NegativeNumbers) {
    vector<int> nums = {-3, -2, -1, -5, -6, -4};
    int value;
    EXPECT_EQ(FindKthLargest(nums, 1, value), 0);
    EXPECT_EQ(value, -1);
    EXPECT_EQ(FindKthLargest(nums, 2, value), 0);
    EXPECT_EQ(value, -2);
    EXPECT_EQ(FindKthLargest(nums, 3, value), 0);
    EXPECT_EQ(value, -3);
}

TEST(FindKthLargestTest, MixedPositiveAndNegativeNumbers) {
    vector<int> nums = {3, -2, -1, 5, -6, 4};
    int value;
    EXPECT_EQ(FindKthLargest(nums, 1, value), 0);
    EXPECT_EQ(value, 5);
    EXPECT_EQ(FindKthLargest(nums, 2, value), 0);
    EXPECT_EQ(value, 4);
    EXPECT_EQ(FindKthLargest(nums, 3, value), 0);
    EXPECT_EQ(value, 3);
}

