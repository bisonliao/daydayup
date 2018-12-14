package com.example.PCMPlayer;

import android.os.Handler;
import android.os.Message;

/**
 * Created by bisonliao on 2018/12/13.
 */
public class ThreadWritePCM implements Runnable {

    private SimplePCMPlayer mPlayer=null;
    private Handler mHandler=null;
    private short[] mBuffer = null;
    private int mBufferLen = 0;
    public ThreadWritePCM(SimplePCMPlayer player, Handler handler, short[] buffer, int bufferlen)
    {
        mPlayer = player;
        mHandler = handler;
        mBuffer = buffer;
        mBufferLen = bufferlen;
    }
    @Override
    public void run() {
/*
        int i;
        int sampleNum = MainActivity.SAMPLE_NUM;


        short[] samples = new short[sampleNum];
        for (i = 0; i < sampleNum; ++i)
        {
            samples[i]  = (short) (Math.sin(2*3.14159265 /44100 * i) * Short.MAX_VALUE);

        }




        mPlayer.write(samples, 0, sampleNum);
        */
        mPlayer.write(mBuffer, 0, mBufferLen);

        Message msg = new Message();
        msg.what = R.integer.WR_OVER;
        mHandler.sendMessage(msg);
    }


}
