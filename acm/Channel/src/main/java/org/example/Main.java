package org.example;

import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.SocketAddress;

public class Main {
    public static void main(String[] args) throws Exception
    {

        if (args.length > 2)
        {
            // ip port asserver
            InetAddress address = InetAddress.getByName(args[0]);  //目标主机地址，这里发送到本机所以是127.0.0.1
            int port = Integer.valueOf(args[1]).intValue();

            DatagramSocket socket = new DatagramSocket(null);
            socket.bind(new InetSocketAddress(address, port));

            Thread thread = new Thread(new Receiver(socket));
            thread.start();
            System.out.println("server continue...");

        }
        else if (args.length == 2)
        {
            // ip port
            InetAddress address = InetAddress.getByName(args[0]);  //目标主机地址，这里发送到本机所以是127.0.0.1
            int port = Integer.valueOf(args[1]).intValue();

            DatagramSocket socket = new DatagramSocket();
            socket.connect(new InetSocketAddress(address, port));
            Sender sender = new Sender(socket);

            Thread thread = new Thread(sender);
            thread.start();
            System.out.println("client continue...");
            int i = 0;
            while(true)
            {
                Thread.sleep(1);
                synchronized (sender)
                {
                    for (int j = 0; j < 2; ++j) {
                        sender.newEvents.add(("hello" + i).getBytes());
                        i++;
                    }
                }
            }

        }

    }
}