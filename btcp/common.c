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
        srandom(time(NULL));
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
    printf("%s ack_seq:%u, dest port:%d, source port:%d, seq:%u, data offset:%d, windowsz:%d, SYNC=%d, ACK=%d\n", 
        msg,
        ntohl(hdr->ack_seq), 
        ntohs(hdr->dest), 
        ntohs(hdr->source), 
        ntohl(hdr->seq), 
        btcp_get_tcphdr_offset(&hdr->doff_res_flags), 
        ntohs(hdr->window),
        btcp_check_tcphdr_flag(FLAG_SYN, hdr->doff_res_flags),
        btcp_check_tcphdr_flag(FLAG_ACK, hdr->doff_res_flags)
        );
    return 0;
}
    



