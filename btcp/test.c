#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>

int get_route_mtu(const char *dest_ip) {
    int sock;
    struct {
        struct nlmsghdr nh;
        struct rtmsg rt;
        char buffer[4096];
    } req;
    struct sockaddr_nl nladdr = {0};
    struct nlmsghdr *nh;
    struct rtmsg *rtm;
    struct rtattr *rta;
    int mtu = -1;

    // 创建 netlink 套接字
    sock = socket(AF_NETLINK, SOCK_DGRAM, NETLINK_ROUTE);
    if (sock < 0) {
        perror("Netlink socket creation failed");
        return -1;
    }

    // 准备请求
    memset(&req, 0, sizeof(req));
    req.nh.nlmsg_len = NLMSG_LENGTH(sizeof(struct rtmsg));
    req.nh.nlmsg_type = RTM_GETROUTE;
    req.nh.nlmsg_flags = NLM_F_REQUEST | NLM_F_DUMP;
    req.rt.rtm_family = AF_INET;

    // 发送请求
    if (send(sock, &req, req.nh.nlmsg_len, 0) < 0) {
        perror("Netlink send failed");
        close(sock);
        return -1;
    }

    // 接收响应
    while (1) {
        int len = recv(sock, &req, sizeof(req), 0);
        if (len < 0) {
            perror("Netlink receive failed");
            break;
        }

        // 遍历响应中的路由信息
        for (nh = (struct nlmsghdr *)&req; NLMSG_OK(nh, len); nh = NLMSG_NEXT(nh, len)) {
            if (nh->nlmsg_type == NLMSG_DONE) {
                goto out;
            }
            if (nh->nlmsg_type == NLMSG_ERROR) {
                fprintf(stderr, "Netlink error\n");
                goto out;
            }

            rtm = NLMSG_DATA(nh);
            rta = RTM_RTA(rtm);
            int rta_len = RTM_PAYLOAD(nh);

            for (; RTA_OK(rta, rta_len); rta = RTA_NEXT(rta, rta_len)) {
                if (rta->rta_type == RTA_OIF) { // 获取接口索引
                    int if_index = *(int *)RTA_DATA(rta);
                    mtu = if_index; // 假设获取成功
                    goto out;
                }
            }
        }
    }

out:
    close(sock);
    return mtu;
}

int main() {
    const char *dest_ip = "8.8.8.8"; // 替换为目标地址
    int mtu = get_route_mtu(dest_ip);
    if (mtu > 0) {
        printf("MTU for destination %s: %d bytes\n", dest_ip, mtu);
    } else {
        printf("Failed to get MTU.\n");
    }
    return 0;
}

