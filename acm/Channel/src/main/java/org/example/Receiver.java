/*
发送端的实现
 */

package org.example;

import javax.swing.*;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashSet;

public class Receiver implements Runnable{

    private DatagramSocket m_socket; // 通信的udp socket

    public Receiver(DatagramSocket socket)
    {
        m_socket = socket;
    }




    @Override
    public void run() {
        byte[] buf = new byte[64*1024];
        System.out.println("begin thread...");
        while (true)
        {
            try
            {
                DatagramPacket p = new DatagramPacket(buf, buf.length);

                m_socket.receive(p);
                //System.out.println("received a package, len="+p.getLength());
                if (p.getLength() < 1)
                {
                    continue;
                }
                byte[] validbytes = new byte[p.getLength()];
                for (int i = 0; i < p.getLength(); ++i)
                {
                    validbytes[i] = p.getData()[i];
                }
                Message.Request request = Message.Request.parseFrom(validbytes);
                if (request.getCmd() == 1) // send event
                {
                    Message.SendEvent se = Message.SendEvent.parseFrom(request.getBody().toByteArray());
                    // respond it with a AckEvent Message
                    Message.AckEvent ae = Message.AckEvent.newBuilder().setSequence(se.getSequence()).build();
                    Message.Request req = Message.Request.newBuilder()
                            .setCmd(2)
                            .setBody(com.google.protobuf.ByteString.copyFrom(ae.toByteArray())).build();
                    byte[] r = req.toByteArray();
                    DatagramPacket response = new DatagramPacket(r, r.length);
                    response.setAddress(p.getAddress());
                    response.setPort(p.getPort());
                    this.m_socket.send(response);
                }
            }
            catch(Exception e)
            {
                e.printStackTrace();
            }

        }
    }
}
