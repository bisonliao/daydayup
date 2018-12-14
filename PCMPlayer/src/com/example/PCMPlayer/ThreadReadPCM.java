package com.example.PCMPlayer;

import android.os.Handler;
import android.os.Message;

/**
 * Created by bisonliao on 2018/12/14.
 */
public class ThreadReadPCM implements Runnable {
    private SimplePCMRecorder mRecorder = null;
    private Handler mHandler = null;
    private short[] mBuffer = null;
    private int mBufferLen = 0;

    public ThreadReadPCM(SimplePCMRecorder recorder, Handler handler, short[] buffer, int bufferlen)
    {
        mHandler = handler;
        mRecorder = recorder;
        mBuffer = buffer;
        mBufferLen = bufferlen;
    }


    @Override
    public void run() {

        int len = mRecorder.read(mBuffer, mBufferLen);
        Message msg = new Message();
        msg.what = 23;
        msg.arg1 = len;
        mHandler.sendMessage(msg);

    }
}
