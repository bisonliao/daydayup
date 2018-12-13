package com.bison.ListenEnglish;

import android.os.Handler;
import android.os.Message;

/**
 * Created by Administrator on 2015/12/23.
 */
public class ThreadMonitorTime implements Runnable {
    
    Handler m_handler;

    public ThreadMonitorTime(Handler handler)
    {
        m_handler = handler;
    }

    @Override
    public void run() {
        while (true)
        {
            try {
                Thread.sleep(100);
                Message msg = new Message();
                msg.what = R.integer.MSGID_CHECK_TIME;
                m_handler.sendMessage(msg);
            }
            catch (Exception e)
            {
                return;
            }

        }
    }
}
