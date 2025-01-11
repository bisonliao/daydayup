#include "btcp_rtt.h"

// 统计rtt的实现

int btcp_rtt_init(struct btcp_rtt_handler * rtt_handler)
{
    rtt_handler->current_rtt_msec = 300;
    rtt_handler->send_record_list = NULL;
    return 0;
}
static void btcp_rtt_truncate_list(struct btcp_rtt_handler * rtt_handler)
{
   
    GList *iter = rtt_handler->send_record_list;
    
    int counter = 0;
    while ( iter != NULL)
    {
        struct btcp_rtt_send_record *rec = (struct btcp_rtt_send_record *)iter->data;
        if (counter > 100) 
        {
            
            //删除当前元素，有点小技巧
            GList * next = iter->next;
            rtt_handler->send_record_list = g_list_delete_link(rtt_handler->send_record_list, iter); 
            free(rec);
            iter = next;  //保证iter还有效
        }
        else
        {
            iter = iter->next;
            counter++;
        }
    }
    return;
}
int btcp_rtt_add_send_record(struct btcp_rtt_handler * rtt_handler, uint32_t ack_seq_expected)
{
    struct btcp_rtt_send_record *rec = (struct btcp_rtt_send_record *)malloc(sizeof(struct btcp_rtt_send_record));
    if (rec == NULL)
    {
        return -1;
    }
    rec->ack_seq = ack_seq_expected;
    rec->sent_msec = btcp_get_monotonic_msec();
    rtt_handler->send_record_list = g_list_insert(rtt_handler->send_record_list, rec, 0);

    btcp_rtt_truncate_list(rtt_handler);
    return 0;
}
int btcp_rtt_update_rtt(struct btcp_rtt_handler * rtt_handler, uint32_t ack_seq)
{
    GList * iter = NULL;
    for (iter = rtt_handler->send_record_list; iter != NULL; iter = iter->next)
    {
        struct btcp_rtt_send_record *rec = (struct btcp_rtt_send_record *)iter->data;
        if (rec->ack_seq == ack_seq)
        {
            uint64_t current = btcp_get_monotonic_msec();
            if (current > rec->sent_msec)
            {
                uint64_t one_rtt = current - rec->sent_msec;
                if (one_rtt < 1000)// rtt 大于1s应该是噪声
                {
                    rtt_handler->current_rtt_msec = rtt_handler->current_rtt_msec * 0.8 + one_rtt * 0.2;
                    g_info("in rtt, update to %u", rtt_handler->current_rtt_msec);
                }
                
            }
            break;
        }
    }
    return 0;
}
int btcp_rtt_destroy(struct btcp_rtt_handler * rtt_handler)
{
    for (const GList *iter = rtt_handler->send_record_list; iter != NULL; iter = iter->next) {
        struct btcp_rtt_send_record *rec = (struct btcp_rtt_send_record *)iter->data;
        free(rec);
    }
    g_list_free(rtt_handler->send_record_list);  // 释放链表
    rtt_handler->send_record_list = NULL;
    return 0;
}

#if 0
// 单元测试代码

int main()
{
    struct btcp_rtt_handler rtt_handler;

    // 1. 初始化 RTT 处理器
    if (btcp_rtt_init(&rtt_handler) != 0) {
        fprintf(stderr, "Failed to initialize RTT handler\n");
        return 1;
    }
    printf("RTT handler initialized. Initial RTT: %u ms\n", rtt_handler.current_rtt_msec);

    // 2. 模拟发送报文并记录发送时间
    uint32_t ack_seq_expected = 1001; // 假设第一个报文的 ACK 序号是 1001
    if (btcp_rtt_add_send_record(&rtt_handler, ack_seq_expected) != 0) {
        fprintf(stderr, "Failed to add send record\n");
        return 1;
    }
    printf("Added send record for ACK seq: %u\n", ack_seq_expected);

    // 模拟网络延迟
    sleep(1);

    // 3. 模拟接收 ACK 并更新 RTT
    if (btcp_rtt_update_rtt(&rtt_handler, ack_seq_expected) != 0) {
        fprintf(stderr, "Failed to update RTT\n");
        return 1;
    }
    printf("Updated RTT. Current RTT: %u ms\n", rtt_handler.current_rtt_msec);

    // 4. 发送更多报文并更新 RTT
    for (int i = 0; i < 5; i++) {
        ack_seq_expected++;
        if (btcp_rtt_add_send_record(&rtt_handler, ack_seq_expected) != 0) {
            fprintf(stderr, "Failed to add send record\n");
            return 1;
        }
        printf("Added send record for ACK seq: %u\n", ack_seq_expected);

        // 模拟网络延迟
        usleep(500000); // 500ms

        if (btcp_rtt_update_rtt(&rtt_handler, ack_seq_expected) != 0) {
            fprintf(stderr, "Failed to update RTT\n");
            return 1;
        }
        printf("Updated RTT. Current RTT: %u ms\n", rtt_handler.current_rtt_msec);
    }

    // 5. 测试链表截断功能
    // 发送大量报文以触发链表截断
    for (int i = 0; i < 200; i++) {
        ack_seq_expected++;
        if (btcp_rtt_add_send_record(&rtt_handler, ack_seq_expected) != 0) {
            fprintf(stderr, "Failed to add send record\n");
            return 1;
        }
    }
    printf("Added 200 send records. List should be truncated. len=%d\n", g_list_length(rtt_handler.send_record_list));

    // 6. 清理资源
    if (btcp_rtt_destroy(&rtt_handler) != 0) {
        fprintf(stderr, "Failed to destroy RTT handler\n");
        return 1;
    }
    printf("RTT handler destroyed. Test completed.\n");

    return 0;
}

#endif