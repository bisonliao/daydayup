#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <pthread.h>
#include <time.h>
#include "btcp.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <net/if.h>

#define BUFSIZE (8*1024)

static int get_mtu_from_sysfs(int ifindex) {
    char ifname[1024];
    if (!if_indextoname(ifindex, ifname)) {
        perror("if_indextoname");
        return -1;
    }

    char path[256];
    snprintf(path, sizeof(path), "/sys/class/net/%s/mtu", ifname);

    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror("fopen");
        return -1;
    }

    int mtu;
    if (fscanf(fp, "%d", &mtu) != 1) {
        //fprintf(stderr, "Failed to read MTU from %s\n", path);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    //fprintf(stderr, "MTU from sysfs: %d\n", mtu);
    return mtu;
}

static int get_interface_mtu(int sockfd, int ifindex) {
    struct nlmsghdr *nlh;
    struct ifinfomsg *ifi;
    struct rtattr *rta;
    char buf[BUFSIZE];
    int len;

    struct {
        struct nlmsghdr nh;
        struct ifinfomsg ifi;
    } req;

    memset(&req, 0, sizeof(req));
    req.nh.nlmsg_len = NLMSG_LENGTH(sizeof(struct ifinfomsg));
    req.nh.nlmsg_type = RTM_GETLINK;
    req.nh.nlmsg_flags = NLM_F_REQUEST;
    req.nh.nlmsg_seq = 1;
    req.ifi.ifi_family = AF_UNSPEC;
    req.ifi.ifi_index = ifindex;

    //fprintf(stderr, "Sending request to get interface MTU for ifindex: %d\n", ifindex);

    if (send(sockfd, &req, req.nh.nlmsg_len, 0) < 0) {
        perror("send RTM_GETLINK");
        return -1;
    }

    len = recv(sockfd, buf, BUFSIZE, 0);
    if (len < 0) {
        perror("recv RTM_GETLINK");
        return -1;
    }

    //fprintf(stderr, "Received response for interface MTU\n");

    for (nlh = (struct nlmsghdr *)buf; NLMSG_OK(nlh, len); nlh = NLMSG_NEXT(nlh, len)) {
        if (nlh->nlmsg_type == NLMSG_DONE) break;
        if (nlh->nlmsg_type == NLMSG_ERROR) {
            //fprintf(stderr, "Error in Netlink response\n");
            return -1;
        }

        ifi = NLMSG_DATA(nlh);
        rta = IFLA_RTA(ifi);
        int rta_len = IFLA_PAYLOAD(nlh);

        for (; RTA_OK(rta, rta_len); rta = RTA_NEXT(rta, rta_len)) {
            //fprintf(stderr, "Attribute type: %d\n", rta->rta_type);
            if (rta->rta_type == IFLA_MTU) {
                int mtu = *(int *)RTA_DATA(rta);
                //fprintf(stderr, "Found MTU: %d\n", mtu);
                return mtu;
            }
        }
    }

    //fprintf(stderr, "MTU not found for ifindex: %d\n", ifindex);
    //fprintf(stderr, "Falling back to sysfs for MTU\n");
    return  get_mtu_from_sysfs(ifindex);
   
}

int btcp_get_route_mtu(const char *dest_ip) {
    int sockfd = socket(AF_NETLINK, SOCK_RAW, NETLINK_ROUTE);
    if (sockfd < 0) {
        perror("socket");
        return -1;
    }

    //fprintf(stderr, "Socket created for Netlink communication\n");

    struct {
        struct nlmsghdr nh;
        struct rtmsg rt;
        char buf[BUFSIZE];
    } req;

    memset(&req, 0, sizeof(req));

    req.nh.nlmsg_len = NLMSG_LENGTH(sizeof(struct rtmsg));
    req.nh.nlmsg_type = RTM_GETROUTE;
    req.nh.nlmsg_flags = NLM_F_REQUEST | NLM_F_DUMP;
    req.nh.nlmsg_seq = 1;

    req.rt.rtm_family = AF_INET;
    req.rt.rtm_table = RT_TABLE_MAIN;

    struct rtattr *rta = (struct rtattr *)req.buf;
    rta->rta_type = RTA_DST;
    rta->rta_len = RTA_LENGTH(4);
    if (inet_pton(AF_INET, dest_ip, RTA_DATA(rta)) != 1) {
        //fprintf(stderr, "Invalid destination IP: %s\n", dest_ip);
        close(sockfd);
        return -1;
    }
    req.nh.nlmsg_len += rta->rta_len;

    //fprintf(stderr, "Sending Netlink request for route to destination: %s\n", dest_ip);

    if (send(sockfd, &req, req.nh.nlmsg_len, 0) < 0) {
        perror("send RTM_GETROUTE");
        close(sockfd);
        return -1;
    }

    char buf[BUFSIZE];
    int len = recv(sockfd, buf, BUFSIZE, 0);
    if (len < 0) {
        perror("recv RTM_GETROUTE");
        close(sockfd);
        return -1;
    }

    //fprintf(stderr, "Received Netlink response for route\n");

    struct nlmsghdr *nlh;
    for (nlh = (struct nlmsghdr *)buf; NLMSG_OK(nlh, len); nlh = NLMSG_NEXT(nlh, len)) {
        if (nlh->nlmsg_type == NLMSG_DONE) break;
        if (nlh->nlmsg_type == NLMSG_ERROR) {
            //fprintf(stderr, "Error in Netlink response\n");
            close(sockfd);
            return -1;
        }

        struct rtmsg *rtm = NLMSG_DATA(nlh);
        struct rtattr *attr = RTM_RTA(rtm);
        int attr_len = RTM_PAYLOAD(nlh);

        int ifindex = -1;
        for (; RTA_OK(attr, attr_len); attr = RTA_NEXT(attr, attr_len)) {
            if (attr->rta_type == RTA_OIF) {
                ifindex = *(int *)RTA_DATA(attr);
                //fprintf(stderr, "Found interface index: %d\n", ifindex);
            }
        }
        

        if (ifindex != -1) {
            int mtu = get_interface_mtu(sockfd, ifindex);
            
            close(sockfd);
            return mtu;
        }
    }

    //fprintf(stderr, "No route found for destination IP: %s\n", dest_ip);
    close(sockfd);
    return -1;
}


int btcp_errno;
static unsigned char bmp[65536 / 8 - 1024 / 8];
static pthread_mutex_t bmp_mutex = PTHREAD_MUTEX_INITIALIZER;
int btcp_alloc_local_port()
{
    int i, j;
    pthread_mutex_lock(&bmp_mutex);
    for (i = 0; i < sizeof(bmp); ++i)
    {
        if (bmp[i] != 255)
        {
            for (j = 0; j < 8; ++j)
            {
                if ( (bmp[i] & (1 << j)) == 0)
                {
                    bmp[i] = bmp[i] | (1 << j);
                    pthread_mutex_unlock(&bmp_mutex);
                    return (i*8+j)+1024;
                }
            }
        }
    }
    pthread_mutex_unlock(&bmp_mutex);
    return -1;
}
int btcp_free_local_port(unsigned short port)
{
    int i, j;
    if (port < 1024 ) { return -1;}
    port = port - 1024;
    i = port / 8 ;
    j = port % 8;

    pthread_mutex_lock(&bmp_mutex);
    if ( (bmp[i] & (1 << j)) != 0)
    {

        bmp[i] = bmp[i] & (~(1 << j));
    }
    pthread_mutex_unlock(&bmp_mutex);
    return 0;
}

unsigned int btcp_get_random() 
{
    // 初始化随机数种子
    static char initflag = 0;
    if (!initflag)
    {
        srandom(getpid());
        initflag = 1;
    }
    // 生成 10 个随机数
    int v = random(); // 生成一个随机数
    return *(unsigned int *)&v;
    
}

int btcp_set_tcphdr_flag(enum btcp_tcphdr_flag flag, uint16_t * doff_res_flags)
{
    uint16_t v = 1 << ((int)flag);

    *doff_res_flags = ntohs(*doff_res_flags);
    *doff_res_flags = (*doff_res_flags) | v;
    *doff_res_flags = htons(*doff_res_flags);

    return 0;
}
int btcp_clear_tcphdr_flag(enum btcp_tcphdr_flag flag, uint16_t * doff_res_flags)
{
    uint16_t v = 1 << ((int)flag);
    v = ~v;

    *doff_res_flags = ntohs(*doff_res_flags);
    *doff_res_flags = (*doff_res_flags) & v;
    *doff_res_flags = htons(*doff_res_flags);

    return 0;
}
int btcp_check_tcphdr_flag(enum btcp_tcphdr_flag flag, uint16_t doff_res_flags)
{
    uint16_t v = 1 << ((int)flag);
    doff_res_flags = ntohs(doff_res_flags);
    if (doff_res_flags  & v)
    {
        return 1;
    }
    return 0;
}
/*
字段布局 (TCP Offset, Reserved, Flags)， 对应主机上一个uint16_t的各个位如下：
| 位索引 (Bit Index) | 15 - 12 (4 位) | 11 - 9 (3 位)  | 8 - 0 (9 位)      |
|--------------------|----------------|----------------|-------------------|
| 字段               | Offset         | Reserved       | Flags            |

字段说明：
- Offset  : 表示 TCP 首部长度，单位是 4 字节。
- Reserved: 预留字段，通常置 0，用于未来扩展。
- Flags   : TCP 控制位，包括 ACK, SYN, FIN 等标志。

*/
int btcp_set_tcphdr_offset(int offset, uint16_t * doff_res_flags)
{
    if (offset > 60 || offset < 0) {return -1;}
    uint16_t off = offset;

    off = off / 4;
    

    *doff_res_flags = ntohs(*doff_res_flags);
    
    // 清除该字段的 高 4 位
    *doff_res_flags &= 0x0FFF;
    // 将 off 写入 该字段 的高 4 位
    *doff_res_flags |= (off << 12);
    *doff_res_flags = htons(*doff_res_flags);

}
int btcp_get_tcphdr_offset(const uint16_t * doff_res_flags)
{
    uint16_t offset = ntohs(*doff_res_flags);
    offset = (offset >> 12) & 0x0f;
    return offset * 4;
}
// 设置非阻塞
int btcp_set_socket_nonblock(int sockfd)
{
    int flags = fcntl(sockfd, F_GETFL, 0);
    fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
    return 0;
}

int btcp_is_readable(int sockfd, int to, char * bigbuffer, int buflen, struct sockaddr_in *client_addr)
{
    struct pollfd pfd[1];
    pfd[0].fd = sockfd;
    pfd[0].events = POLLIN;
    
    
    socklen_t addr_len = sizeof(struct sockaddr);

    int ret = poll(pfd, 1, to);
    if (ret > 0)
    {
        if (pfd[0].revents & POLLIN)
        {
            ssize_t received = recvfrom(pfd[0].fd, bigbuffer, buflen, 0,
                                        (struct sockaddr *)client_addr, &addr_len);
            if (received >= 0)
            {
                char ip_str[INET_ADDRSTRLEN]; // 用于存储IP地址字符串
                inet_ntop(AF_INET, &client_addr->sin_addr, ip_str, INET_ADDRSTRLEN);
            
                printf("recv %d bytes from %s\n", received, ip_str);
                return received;
            }
            else if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                return 0;
            }
            else
            {
                return -1;
            }
        }
    }
}
int btcp_get_port(const char*bigbuffer, unsigned short * dest, unsigned short *source)
{
    if (bigbuffer == NULL) {return -1;}
    union btcp_tcphdr_with_option *tcphdr = (union btcp_tcphdr_with_option *)bigbuffer;
    struct btcp_tcphdr * hdr = &tcphdr->base_hdr;
    *dest = ntohs(hdr->dest);
    *source = ntohs(hdr->source);
    return 0;
}

int btcp_print_tcphdr(const char*bigbuffer, const char * msg)
{
    union btcp_tcphdr_with_option *tcphdr = (union btcp_tcphdr_with_option *)bigbuffer;
    struct btcp_tcphdr * hdr = &tcphdr->base_hdr;
    g_info("%s ack_seq:%u, dest port:%d, source port:%d, seq:%u, data offset:%d, windowsz:%d, SYNC=%d, ACK=%d, FIN=%d,\n", 
        msg,
        ntohl(hdr->ack_seq), 
        ntohs(hdr->dest), 
        ntohs(hdr->source), 
        ntohl(hdr->seq), 
        btcp_get_tcphdr_offset(&hdr->doff_res_flags), 
        ntohs(hdr->window),
        btcp_check_tcphdr_flag(FLAG_SYN, hdr->doff_res_flags),
        btcp_check_tcphdr_flag(FLAG_ACK, hdr->doff_res_flags),
        btcp_check_tcphdr_flag(FLAG_FIN, hdr->doff_res_flags)
        );
    return 0;
}

int btcp_check_udp_port_in_use(unsigned short port) 
{
    int sockfd;
    struct sockaddr_in addr;

    // 创建 UDP socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd == -1) {
        return -1;
    }

    // 设置地址结构
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY); // 绑定到所有接口
    addr.sin_port = htons(port);              // 绑定到指定端口

    // 尝试绑定
    if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        if (errno == EADDRINUSE) {
            close(sockfd);
            return 1; // 端口已被占用
        } else {
            close(sockfd);
            return -1; // 发生其他错误
        }
    }

    // 绑定成功，端口未被占用
    close(sockfd);
    return 0;
}

uint32_t btcp_sequence_round_in(uint64_t original)
{
    uint64_t seq64 = original;
    seq64 =  seq64 %((uint64_t)1+UINT32_MAX);
    uint32_t seq32 = seq64;
    return seq32;
}

uint64_t btcp_sequence_round_out(uint32_t original)
{
     return ((uint64_t)original) + UINT32_MAX + 1;
}

uint32_t btcp_sequence_step_forward(uint32_t original, uint32_t steps)
{
    uint64_t tmp_seq = original;
    tmp_seq += steps;
    return tmp_seq % ((uint64_t)1 + UINT32_MAX);
}
uint32_t btcp_sequence_step_back(uint32_t original, uint32_t steps)
{
    if (original < steps) // 不够减
    {
        uint64_t result =  (uint64_t)original + UINT32_MAX + 1 - steps;
        return btcp_sequence_round_in(result);
    }
    else
    {
        return original - steps;
    }
    uint64_t tmp_seq = original;
    tmp_seq += steps;
    return tmp_seq % ((uint64_t)1 + UINT32_MAX);
}

uint64_t btcp_get_monotonic_msec()
{
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);

    uint64_t result = (uint64_t)tp.tv_sec * 1000;
    result += tp.tv_nsec / 1000000;

    return  result;
} 

// 比较两个 range 的起始位置
static int btcp_range_compare(const struct btcp_range *r1, const struct btcp_range *r2) {
    if (r1->from < r2->from) return -1;
    if (r1->from > r2->from) return 1;
    return 0;
}

// 检查两个 range 是否重叠
int btcp_range_overlap(const struct btcp_range *r1, const struct btcp_range *r2) {
    return (r1->from <= r2->to && r1->to >= r2->from);
}

// 从单个 range 中减去另一个 range
static GList *btcp_range_subtract_single(const struct btcp_range *a, const struct btcp_range *b) {
    GList *result = NULL;

    //printf("a:[%llu, %llu], b:[%llu, %llu]\n", a->from, a->to, b->from, b->to);

    if (!btcp_range_overlap(a, b)) {
        // 如果没有重叠，直接返回 a
        struct btcp_range *new_range = malloc(sizeof(struct btcp_range));
        new_range->from = a->from;
        new_range->to = a->to;
        result = g_list_append(result, new_range);
        return result;
    }

    // 处理重叠部分
    if (a->from < b->from) {
        // 左边剩余部分
        struct btcp_range *left = malloc(sizeof(struct btcp_range));
        left->from = a->from;
        left->to = b->from - 1;
        //printf("left:[%llu, %llu]\n", left->from, left->to);
        result = g_list_append(result, left);
    }

    if (a->to > b->to) {
        // 右边剩余部分
        struct btcp_range *right = malloc(sizeof(struct btcp_range));
        right->from = b->to + 1;
        right->to = a->to;
        //printf("right:[%llu, %llu]\n", right->from, right->to);
        result = g_list_append(result, right);
    }

    return result;
}


// 深度拷贝的时候用的元素拷贝函数
static gpointer copy_range(gconstpointer src, gpointer user_data) {
    const struct btcp_range *original = (const struct btcp_range *)src;
    struct btcp_range *a1 = malloc(sizeof(struct btcp_range));
    a1->from = original->from;
    a1->to = original->to;
    return a1;
}
// 从一组 range 中减去另一组 range
int btcp_range_subtract( GList *a,  GList *b, GList **result) {
    GList *tmp_result = NULL;
    GList *aa = NULL;
    aa = g_list_copy_deep(a, copy_range, NULL);
   

    // 遍历 b 中的每个 range
    for (const GList *b_iter = b; b_iter != NULL; b_iter = b_iter->next)
    {
        struct btcp_range *b_range = (struct btcp_range *)b_iter->data;
        GList * new_a = NULL;

        // 遍历 a 中的每个 range
        for (const GList *a_iter = aa; a_iter != NULL; a_iter = a_iter->next)
        {
            struct btcp_range *a_range = (struct btcp_range *)a_iter->data;

            GList *subtracted = btcp_range_subtract_single(a_range, b_range);
            new_a = g_list_concat(new_a, subtracted); //subtracted这时候就变成了new_a的一部分，可以认为无效了，不要释放，也不要做其他操作

#if 0
            if (btcp_range_overlap(a_range, b_range)) // 有重叠就减去
            {
                GList *subtracted = btcp_range_subtract_single(a_range, b_range);
                new_a = g_list_concat(new_a, subtracted); //subtracted这时候就变成了new_a的一部分，可以认为无效了，不要释放，也不要做其他操作
            }
            else // 没有重叠就原封未动的深度拷贝这个元素到结果list
            {
                struct btcp_range *a1 = malloc(sizeof(struct btcp_range));
                a1->from = a_range->from;
                a1->to = a_range->to;
                new_a = g_list_append(new_a, a1); 
            }
#endif
        }
        // 处理下一个b_range的时候，就用new_a替代aa。因为aa里的每个range都减去了b_range，得到一个新的aa了
        //if (new_a != NULL)
        {
            btcp_range_free_list(aa);
            aa = new_a;
            //printf("get new a: ");
            //btcp_range_print_list(aa);
        }
    }

    *result = aa;
    return 0;
}


// 合并一组 range
int btcp_range_list_combine(GList *a, GList **result) {
    if (a == NULL) {
        *result = NULL;
        return 0;
    }

    // 将 GList 转换为数组并排序
    int count = g_list_length(a);
    struct btcp_range *ranges = malloc(count * sizeof(struct btcp_range));
    int i = 0;
    for (GList *iter = a; iter != NULL; iter = iter->next) {
        ranges[i++] = *(struct btcp_range *)iter->data;
    }
    qsort(ranges, count, sizeof(struct btcp_range), 
            (int (*)(const void *, const void *))btcp_range_compare);

    // 合并重叠的 range
    GList *combined = NULL;
    struct btcp_range current = ranges[0];

    for (i = 1; i < count; i++) {
        if (ranges[i].from <= current.to + 1) {
            // 如果当前 range 和下一个 range 重叠或相邻，合并它们
            if (ranges[i].to > current.to) {
                current.to = ranges[i].to;
            }
        } else {
            // 如果不重叠，将当前 range 添加到结果中
            struct btcp_range *new_range = malloc(sizeof(struct btcp_range));
            *new_range = current;
            combined = g_list_append(combined, new_range);

            // 更新当前 range
            current = ranges[i];
        }
    }

    // 添加最后一个 range
    struct btcp_range *new_range = malloc(sizeof(struct btcp_range));
    *new_range = current;
    combined = g_list_append(combined, new_range);

    // 释放临时数组
    free(ranges);

    // 返回结果
    *result = combined;
    return 0;
}


// 打印 range 列表
void btcp_range_print_list(const GList *list) {
    for (const GList *iter = list; iter != NULL; iter = iter->next) {
        struct btcp_range *range = (struct btcp_range *)iter->data;
        printf("[%llu, %llu] ", range->from, range->to);
    }
    printf("\n");
}

// 释放 range 列表
void btcp_range_free_list(GList *list) {
    for (const GList *iter = list; iter != NULL; iter = iter->next) {
        struct btcp_range *range = (struct btcp_range *)iter->data;
        free(range);
    }
    g_list_free(list);  // 释放链表
}

int btcp_range_equal(const void *a, int a_len, const void *b, int b_len)
{
    struct btcp_range * aa = (struct btcp_range * )a;
    struct btcp_range * bb = (struct btcp_range * )b;
    if (aa->from == bb->from && aa->to == bb->to)
    {
        return 0;
    }
    return aa->from - bb->from;
}




