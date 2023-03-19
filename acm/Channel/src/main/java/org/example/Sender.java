/*
发送端的实现
 */
package org.example;


import java.io.IOException;
import java.net.*;
import java.nio.channels.DatagramChannel;
import java.nio.channels.ServerSocketChannel;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

import com.google.protobuf.ByteString;

public class Sender implements Runnable {
    private DatagramSocket m_socket = null; // 通信用的udp socket
    private long m_sequence = 0; // 发送端的sequence
    private long m_seqToAck = 0; //接收端已经确认的sequence，逐一确认
    private HashMap<Long, byte[]> m_buffer = null; // 事件的缓存

    public ArrayList<byte[]> newEvents = new ArrayList<>();

    private HashSet<Long> sendAgain = new HashSet<Long>(); // 用来debug看看重复发送的情况多不多

    private void checkSendRepeated(long seq) //检查是否有重复发送，看看发送效能
    {
        if (sendAgain.contains(Long.valueOf(seq)))
        {
            System.out.println("重复发送");
        }
        else
        {
            sendAgain.add(Long.valueOf(seq));
        }
    }

    public Sender(DatagramSocket socket)
    {
        this.m_socket = socket;
        this.m_sequence = 0;
        this.m_buffer = new HashMap<Long, byte[]>();
    }
    public void PushEvent(byte[] event) // 发生了一个event，放到buffer里待发送给接收端
    {
        this.m_buffer.put(Long.valueOf(this.m_sequence), event);
        m_sequence++;
    }


    @Override
    public void run() {
        try {
            m_socket.setSoTimeout(1);
        } catch (Exception e) {e.printStackTrace(); return;}

        byte[] buffer = new byte[1024*64];
        HashSet<Long> ackSeqSaved = new HashSet<Long>(); //保存短期内已经收到的ack sequence
        HashMap<Long, Long> sendSeqSaved = new HashMap<>(); // 保存短期内已经发送过的sequence，避免重复发送，超过5s会清理掉。
        while (true)
        {
            //try { Thread.sleep(1000);}catch (Exception e){}
            synchronized (this)
            {
                int i;
                for (i = 0; i < newEvents.size(); ++i)
                {
                    this.PushEvent(newEvents.get(i));
                }
                newEvents.clear();
            }

            try {
                // send events those have NOT been acked
                if (m_sequence > m_seqToAck) {
                    long seq;
                    long now = System.currentTimeMillis();
                    for (seq = m_seqToAck; seq < m_sequence; ++seq) {
                        if (sendSeqSaved.containsKey(Long.valueOf(seq))) //过去5s内有发送过，则不发
                        {
                            continue;
                        }
                        checkSendRepeated(seq);//如果有重复发送，就会在标准输出提示
                        // 构造Send event message并发送给对端
                        Message.SendEvent sendEvent = Message.SendEvent.newBuilder()
                                .setEvent(com.google.protobuf.ByteString.copyFrom(this.m_buffer.get(Long.valueOf(seq))))
                                .setSequence(seq).build();

                        Message.Request request = Message.Request.newBuilder()
                                .setCmd(1)
                                .setBody(com.google.protobuf.ByteString.copyFrom(sendEvent.toByteArray())).build();

                        byte[] r = request.toByteArray();
                        DatagramPacket packet = new DatagramPacket(r, r.length);
                        this.m_socket.send(packet);
                        sendSeqSaved.put(Long.valueOf(seq), Long.valueOf(now));// 记录发送过该seq和发送时间
                    }

                }

                // receive ack message
                DatagramPacket p = new DatagramPacket(buffer, buffer.length);
                m_socket.receive(p);
                if (p.getLength() < 1) {
                    continue;
                }
                byte[] validbytes = new byte[p.getLength()];
                for (int i = 0; i < p.getLength(); ++i) {
                    validbytes[i] = p.getData()[i];
                }
                Message.Request request = Message.Request.parseFrom(validbytes);
                if (request.getCmd() == 2) {
                    Message.AckEvent ae = Message.AckEvent.parseFrom(request.getBody().toByteArray());
                    ackSeqSaved.add(Long.valueOf(ae.getSequence())); //可能收到是乱序的，所以先暂存起来
                }
                // move seqToAck forward, and delete events in buffer
                while (true) {
                    if (ackSeqSaved.contains(Long.valueOf(m_seqToAck))) { // 有收到想要的这个seq的ack消息
                        System.out.println("ack:" + m_seqToAck);
                        m_buffer.remove(Long.valueOf(m_seqToAck)); //清理掉已经发送且收到了ack的事件
                        ackSeqSaved.remove(Long.valueOf(m_seqToAck));//清理掉ack消息的暂存
                        m_seqToAck++;
                    } else {
                        break;
                    }
                }
                // clear the timeout sequence
                long now = System.currentTimeMillis();
                ArrayList<Long> seqToDel = new ArrayList<>();
                for (HashMap.Entry<Long, Long> entry : sendSeqSaved.entrySet()) {
                    Long key = entry.getKey();
                    Long value = entry.getValue();
                    if (value.longValue() < (now - 5000)) {
                        seqToDel.add(key); //不能直接删，否则会出现异常，先保存起来，等会一起删除
                    }
                }
                for (Long key : seqToDel) {
                    sendSeqSaved.remove(key);
                }

            } catch (SocketTimeoutException e) {

            } catch (Exception e) {
                e.printStackTrace();
                return;
            }


        }

    }
}
