package com.bison.ListenEnglish;


import android.app.Activity;
import android.content.SharedPreferences;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.Toast;


import java.io.File;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by Administrator on 2016/5/6.
 */
public class ThreadGetTitles implements Runnable {
    MainActivity mainActivity;
    Handler handler;

    public ThreadGetTitles(MainActivity a, Handler h)
    {
        mainActivity = a;
        handler = h;
    }
    @Override
    public void run() {
     try
     {

         String rootdir = Environment.getExternalStorageDirectory().getAbsolutePath();

         String audiodir = rootdir + "/ListenEnglish";
         File f = new File(audiodir);
         if (!f.exists())
         {
             f.mkdir();
         }
         if (!f.isDirectory())
         {
             throw new Exception(audiodir+" is no a directory");
         }
         String[] filelist = f.list();
         Log.d("", "" +f.list().length);
            mainActivity.m_titles = new String[filelist.length];
         SharedPreferences sp = mainActivity.getSharedPreferences("breakpoints", MainActivity.MODE_PRIVATE);


            for (int i = 0; i < filelist.length; ++i)
            {

                mainActivity.m_titles[i] = filelist[i];

                Set<String> strset = sp.getStringSet(mainActivity.m_titles[i], new HashSet<String>());

                Log.d("bison", "get strin set for "+mainActivity.m_titles[i]+", size:"+strset.size());
                List<Integer> intlist = MainActivity.strSet2IntList(strset);
                mainActivity.m_brkpoints.put(mainActivity.m_titles[i], intlist);

            }


            Message msg = new Message();
            msg.what = R.integer.MSGID_UPDATE_TITLES;
         handler.sendMessage(msg);


        }
        catch (Exception e)
        {
            Log.e("", e.getMessage());
            e.printStackTrace();
            Toast.makeText(mainActivity, e.getMessage(), 1000).show();
            return;
        }

    }
}
