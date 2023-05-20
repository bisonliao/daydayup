// chatGPT生成的暴力计算24点的程序，挺有意思的，就记录下来
#include <stdio.h>

void calculate24(int a, int b, int c, int d) {
    int nums[4] = {a, b, c, d};
    char ops[3] = {'+', '-', '*', '/'};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (j == i) continue;
            for (int k = 0; k < 4; k++) {
                if (k == i || k == j) continue;
                for (int l = 0; l < 4; l++) {
                    if (l == i || l == j || l == k) continue;
                    for (int m = 0; m < 4; m++) {
                        int result = 0;
                        if (m == 0) {
                            result = nums[i] + nums[j];
                        } else if (m == 1) {
                            result = nums[i] - nums[j];
                        } else if (m == 2) {
                            result = nums[i] * nums[j];
                        } else if (m == 3) {
                            if (nums[j] == 0) continue;
                            if (nums[i] % nums[j] != 0) continue;
                            result = nums[i] / nums[j];
                        }
                        for (int n = 0; n < 4; n++) {
                            int temp = result;
                            if (n == 0) {
                                temp += nums[k];
                            } else if (n == 1) {
                                temp -= nums[k];
                            } else if (n == 2) {
                                temp *= nums[k];
                            } else if (n == 3) {
                                if (nums[k] == 0) continue;
                                if (result % nums[k] != 0) continue;
                                temp /= nums[k];
                            }

                            for (int o = 0; o < 4; o++) {
                                int temp2 = temp;
                                if (o == 0) {
                                    temp2 += nums[l];
                                } else if (o == 1) {
                                    temp2 -= nums[l];
                                } else if (o == 2) {
                                    temp2 *= nums[l];
                                } else if (o == 3) {
                                    if (nums[l] == 0) continue;
                                    if (temp % nums[l] != 0) continue;
                                    temp2 /= nums[l];
                                }

                                if (temp2 == 24) {
                                    printf("((%d %c %d) %c %d) %c %d\n", nums[i], ops[m], nums[j], ops[n], nums[k], ops[o], nums[l]);
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    printf("无法构成24点\n");
}

int main() {
    int a, b, c, d;
    printf("请输入4张牌的点数：");
    scanf("%d %d %d %d", &a, &b, &c, &d);
    calculate24(a, b, c, d);
    return 0;
}

