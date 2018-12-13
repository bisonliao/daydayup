package com.bison.ListenEnglish;

import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.widget.Toast;


import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetSocketAddress;
import java.net.Socket;

/**
 * Created by Administrator on 2016/5/6.
 */
public class ThreadGetUrl implements Runnable {
    MainActivity mainActivity;
    Handler handler;

    public ThreadGetUrl(MainActivity a, Handler h)
    {
        mainActivity = a;
        handler = h;
    }
    @Override
    public void run() {


        try {


            mainActivity.m_urlToPlay = Environment.getExternalStorageDirectory()+"/ListenEnglish/"+mainActivity.m_titles[mainActivity.m_currentFilePos];

            Message msg = new Message();
            msg.what = R.integer.MSGID_PLAY_URL;
            handler.sendMessage(msg);


        }
        catch (Exception e)
        {
            e.printStackTrace();
            Toast.makeText(mainActivity, e.getMessage(), 1000).show();
            return;
        }

    }
}
