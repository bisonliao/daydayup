/*
 自然对数e 数字串中第一个连续10个数字构成的素数整数
 bigdecimal是在Pi目录下的工具类
 */


bool isPrime(unsigned long long v)
{

    unsigned long long i;
    for (i = 2; i <= v / 2; ++i)
    {
        if ((v % i) == 0)
        {
            return false;
        }
    }
    return true;
}

int main()
{
    std::cout << "Hello World!\n";

    BigDecimal e;
    BigDecimal v;
    bigdecimal_assign(&e, "2.5");
    bigdecimal_assign(&v, "0.5");


    for (int i = 3; ; ++i)
    {
        BigDecimal result1, result2;
        bigdecimal_div(&v, i, &result1);
        bigdecimal_add(&e, &result1, &result2);

        BigDecimal oldvalue = e;
        e = result2;
        v = result1;

      
        if (memcmp(&e, &oldvalue, sizeof(BigDecimal)) == 0)
        {
            printf("i=%d\n", i);
            break;
        }
        
    }
    static char str[100000];
    bigdecimal_tostring(&e, str);
    printf("%s\n", str);
    printf("len=%d\n", strlen(str));

    unsigned long long maybePrime = 0;
    for (int i = 2; i < 10000; ++i)
    {
        maybePrime = (maybePrime % 1000000000);

        maybePrime = maybePrime * 10;
        maybePrime = maybePrime+(str[i] - '0');
        
        if (i < 10)
        {
            continue;
        }
        
        //printf("i=%d, %llu\n", i, maybePrime);
        if (isPrime(maybePrime))
        {
            printf("i=%d, %llu\n", i, maybePrime);
            break;
        }

    }
    return 0;
}
