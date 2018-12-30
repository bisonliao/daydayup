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


import java.io.*;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;

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
    List<Integer> loadBreakInfo(String filename)
    {
        String rootdir = Environment.getExternalStorageDirectory().getAbsolutePath();

        String audiodir = rootdir + "/ListenEnglish";
        String brkFile = audiodir + "/"+filename+".brk";
        List<Integer> intList = new ArrayList<>();

        try {
            Log.d("bison", "try to open "+brkFile);
            BufferedReader r = new BufferedReader(new FileReader(brkFile));

            String  line = null;

            while ( (line = r.readLine()) != null) {
                intList.add(new Integer(line));
                Log.d("bison", "I read break point:"+line);
            }
            r.close();
        }
        catch (Exception e)
        {
            Log.e("bison", e.getMessage());
        }
        return intList;

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
         Log.d("", "" + f.list().length);
         int mp3fileCnt = 0;
         for (int i = 0; i < filelist.length; ++i) {
             if (filelist[i].length() > 4 && (filelist[i].endsWith(".mp3") || filelist[i].endsWith(".MP3"))) {
                 mp3fileCnt++;
             }
         }
         if (mp3fileCnt < 1)
         {
             return;
         }
         mainActivity.m_titles = new String[mp3fileCnt];
         SharedPreferences sp = mainActivity.getSharedPreferences("breakpoints", MainActivity.MODE_PRIVATE);

            int mp3Index = 0;
            for (int i = 0; i < filelist.length; ++i)
            {
                if (filelist[i].length() > 4 && (filelist[i].endsWith(".mp3")||filelist[i].endsWith(".MP3") ))
                {

                    mainActivity.m_titles[mp3Index++] = filelist[i];
                /*

                    Set<String> strset = sp.getStringSet(mainActivity.m_titles[i], new HashSet<String>());

                    Log.d("bison", "get strin set for " + mainActivity.m_titles[i] + ", size:" + strset.size());
                    List<Integer> intlist = MainActivity.strSet2IntList(strset);
                    mainActivity.m_brkpoints.put(mainActivity.m_titles[i], intlist);
                    */
                    List<Integer> intlist = loadBreakInfo(filelist[i]);
                    mainActivity.m_brkpoints.put(mainActivity.m_titles[i], intlist);
                }

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
