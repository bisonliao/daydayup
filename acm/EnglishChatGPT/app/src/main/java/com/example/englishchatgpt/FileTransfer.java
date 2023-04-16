package com.example.englishchatgpt;
import android.os.Handler;
import android.util.Log;
import android.widget.ArrayAdapter;

import org.java_websocket.WebSocket;
import org.java_websocket.WebSocketImpl;
import org.java_websocket.client.*;
import org.java_websocket.drafts.Draft;
import org.java_websocket.drafts.Draft_6455;
import org.java_websocket.enums.CloseHandshakeType;
import org.java_websocket.enums.HandshakeState;
import org.java_websocket.enums.ReadyState;
import org.java_websocket.exceptions.InvalidDataException;
import org.java_websocket.exceptions.InvalidHandshakeException;
import org.java_websocket.framing.Framedata;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.handshake.ClientHandshakeBuilder;
import org.java_websocket.handshake.HandshakeBuilder;
import org.java_websocket.handshake.ServerHandshake;
import org.java_websocket.handshake.ServerHandshakeBuilder;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.nio.ByteBuffer;
import java.util.Deque;
import java.util.List;

public class FileTransfer implements Runnable{

    private Deque<String> mAudioFileToServer;
    private List<String> mAudioFileFromServer;
    private Handler handler;

    public FileTransfer(Deque<String> audioFileToServer, List<String>  audioFileFromServer, Handler h)
    {
        mAudioFileFromServer = audioFileFromServer;
        mAudioFileToServer = audioFileToServer;
        handler = h;
    }

    @Override
    public void run() {

        String url = "ws://119.28.214.71:7890";


        try {
            WebSocketClient client = new WebSocketClient(new URI(url), new Draft_6455()) {
                @Override
                public void onMessage(ByteBuffer bytes) {
                    super.onMessage(bytes);
                    Log.v("bisonliao", new String(bytes.toString()));
                    if (bytes.toString().length() < 1) { return;}
                    synchronized (FileTransfer.this.mAudioFileFromServer)
                    {
                        FileTransfer.this.mAudioFileFromServer.add(bytes.toString());
                        handler.sendEmptyMessage(0);
                    }

                }

                @Override
                public void onMessage(String message) {

                    Log.v("bisonliao", "recv message:"+message.length());
                    if (message.length() < 1) { return;}
                    synchronized (FileTransfer.this.mAudioFileFromServer)
                    {
                        FileTransfer.this.mAudioFileFromServer.add(message);
                        handler.sendEmptyMessage(0);
                    }
                }

                @Override
                public void onError(Exception ex) {
                    ex.printStackTrace();

                }

                @Override
                public void onOpen(ServerHandshake handshakedata) {
                    System.out.println("connected");

                }

                @Override
                public void onClose(int code, String reason, boolean remote) {

                }
            };
            client.connect();
            //client.run();
            while (!client.getReadyState().equals(ReadyState.OPEN))
            {
                Thread.sleep(10);
            }

            while (true)
            {
                if (!client.getReadyState().equals(ReadyState.OPEN))
                {
                    client.connect();
                }
                String fileName = null;
                // check if any files to send
                synchronized (mAudioFileToServer)
                {
                    if (mAudioFileToServer.size() > 0)
                    {
                       fileName = mAudioFileToServer.getLast();
                       mAudioFileToServer.removeAll(mAudioFileToServer);
                    }
                }

                if (fileName != null)
                {
                    int maxSize = 1024*32;
                    byte[] buffer = new byte[maxSize];
                    FileInputStream r = new FileInputStream(fileName);
                    int len = r.read(buffer);
                    r.close();
                    if (len >= maxSize || len < 1) // file too large
                    {
                        new File(fileName).delete();
                        continue;
                    }
                    // these copy actions are not avoidable ?
                    byte[] buffer2 = new byte[len];
                    for (int i = 0; i < len; ++i)
                    {
                        buffer2[i] = buffer[i];
                    }
                    client.send(buffer2);
                    new File(fileName).delete();
                }

                else
                {
                    Thread.sleep(1000);
                }
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();

        }

    }
}
