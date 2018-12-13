package com.bison.ListenEnglish;


import android.app.Activity;
import android.app.AlertDialog;
import android.app.admin.DeviceAdminReceiver;
import android.content.Context;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.*;
import android.util.Log;
import android.view.View;
import android.widget.*;


import java.io.File;
import java.io.IOException;
import java.net.*;
import java.util.*;

public class MainActivity extends Activity {

    public static  String server_ip = "119.29.151.140";
    public static  String GET_TITLES = "MainLogic.MainLogicService.GetTitles";
    public static  String GET_URL = "MainLogic.MainLogicService.GetUrlByTitle";


    //public static final String server_ip = "119.29.169.48";
    //public static final String GET_TITLES = "MainLogic.MainLogicService.getTitles";
    //public static final String GET_URL = "MainLogic.MainLogicService.getUrlByTitle";
    /**
     * Called when the activity is first created.
     */
    private String m_AudioFileDir = "";
    protected int m_currentFilePos = 0;
    private MediaPlayer m_player = null;
    private boolean m_isPaused = false;
    protected String [] m_titles = null;
    protected  String m_urlToPlay = null;

    private Date m_timeToExit = null;
    protected Map<String,List<Integer> > m_brkpoints = new HashMap<String,List<Integer>>();
    protected int m_lastbrk = 0;//最后一次断句的位置

    private MainActivityHandler m_handler = new MainActivityHandler();
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

      //  checkExternalStorage();
      //  loadFileList();

        ProgressBar pb = (ProgressBar)(MainActivity.this.findViewById(R.id.progressBar));
        pb.setMax(500);

        new Thread(new ThreadMonitorTime(m_handler)).start();//定时触发Handler的handleMessage

        loadFileList();
    }

    @Override
    public void onPostCreate(Bundle savedInstanceState, PersistableBundle persistentState) {
        super.onPostCreate(savedInstanceState, persistentState);

    }

    //从外部存储器的固定目录读取文件列表
    private void loadFileList()
    {

        ListView lv = (ListView)(findViewById(R.id.FileListView));
        lv.setOnItemClickListener(new OnListViewClicked());

        findViewById(R.id.loadingPanel).setVisibility(View.VISIBLE);
        new Thread(new ThreadGetTitles(this ,m_handler)).start();



    }
    //用于处理定时信号的处理器
    class MainActivityHandler extends Handler
    {
        private int m_cnt = -1;

        String millisec2ReadableStr(int millisec)
        {
            int sec = millisec / 1000;
            int min = sec / 60;
            int sec_left = sec - min * 60;
            String ret = ""+min+":"+sec_left;
            return ret;
        }
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            m_cnt++;
            if (msg.what == R.integer.MSGID_CHECK_TIME)
            {
                //更新一下播放时间
                if ( (m_cnt) % 10 == 0)
                {
                    if (m_player != null &&m_player.isPlaying()) {
                        String progressMsg = "" + millisec2ReadableStr(m_player.getCurrentPosition()) + "/" + millisec2ReadableStr(m_player.getDuration());
                        TextView tv = (TextView) (MainActivity.this.findViewById(R.id.TxtVw_Progress));
                        tv.setText(progressMsg);
                    }
                }

                //是不是该退出该app了？
                {
                    Date d = new Date();

                    if (m_timeToExit != null && d.after(m_timeToExit)) {
                        MainActivity.this.finish();
                    }
                }

                //更新进度条
                if (m_player != null &&m_player.isPlaying()) {
                    int max = m_player.getDuration();
                    if (max <= 0) {
                        max = 500;
                    }
                    int cur = m_player.getCurrentPosition();
                    float f_cur = cur;

                    ProgressBar bar = (ProgressBar)(MainActivity.this.findViewById(R.id.progressBar));
                    bar.setProgress((int) (500 * (f_cur / max)));
                }

                //跟读模式，是不是该暂停
                {
                    if (m_player != null &&m_player.isPlaying())
                    {

                        ToggleButton toggleButton = (ToggleButton)(findViewById(R.id.btn_follow));
                        if (toggleButton.isChecked())
                        {
                            int cur = MainActivity.this.m_player.getCurrentPosition();
                            List<Integer> brklist =  MainActivity.this.m_brkpoints.get(MainActivity.this.m_titles[MainActivity.this.m_currentFilePos]);
                            if (brklist != null && brklist.size() > 0)
                            {

                                int brk=-1;
                                if (MainActivity.this.m_lastbrk > cur)// invalid lastbrk
                                {
                                    brk = brklist.get(0).intValue();
                                    //Log.d("bison", "get first break positon:"+brk);
                                }
                                else
                                {
                                    int i;
                                    for (i = 0; i < brklist.size();++i)
                                    {
                                        if (brklist.get(i).intValue() > MainActivity.this.m_lastbrk)
                                        {
                                            brk = brklist.get(i).intValue();
                                           /// Log.d("bison","get #"+i+" value :"+brk);
                                            break;
                                        }
                                    }
                                }
                               /// Log.d("bison", "brk:"+brk+" cur:"+cur);
                                if (brk != -1 && cur > brk)
                                {

                                    MainActivity.this.pauseOrContinue();
                                    MainActivity.this.m_lastbrk = brk;
                                }
                            }


                        }
                    }

                }
            }
            if (msg.what == R.integer.MSGID_UPDATE_TITLES)
            {
                findViewById(R.id.loadingPanel).setVisibility(View.GONE);

                ListView lv = (ListView)(MainActivity.this.findViewById(R.id.FileListView));

                ArrayAdapter<String> ad = new ArrayAdapter<String>(MainActivity.this, android.R.layout.simple_list_item_1, m_titles);
                lv.setAdapter(ad);
            }
            if (msg.what == R.integer.MSGID_PLAY_URL)
            {
                findViewById(R.id.loadingPanel).setVisibility(View.GONE);
                try {
                    m_player.setDataSource(m_urlToPlay);
                    m_player.prepare();
                    m_player.start();


                } catch (Exception e) {
                    Toast.makeText(MainActivity.this, "exception:" + e.getMessage(), 1000).show();
                    return;
                }

                Button btn = (Button)(MainActivity.this.findViewById(R.id.btn_play));
                btn.setText("pause");

                TextView tv = (TextView)(MainActivity.this.findViewById(R.id.txtVw_ShowFileName));
                tv.setText(""+m_currentFilePos+" "+m_titles[m_currentFilePos]);

                //����һ���̣߳����½�����
                ProgressBar bar = (ProgressBar)(MainActivity.this.findViewById(R.id.progressBar));
            }
        }
    }
    //点击ListView开始播放某个文件
    class OnListViewClicked implements AdapterView.OnItemClickListener
    {

        @Override
        public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {

            m_currentFilePos = i;

            PlayCurrentFile();

        }
    }
    //播放完一个文件继续下一个文件的播放
    class  OnPlayCompletion implements MediaPlayer.OnCompletionListener
    {
        @Override
        public void onCompletion(MediaPlayer mediaPlayer) {

            ToggleButton toggleButton = (ToggleButton)(findViewById(R.id.btn_repeat));
            if (toggleButton.isChecked())
            {
                PlayCurrentFile();
            } else {

                if (m_currentFilePos < (m_titles.length - 1)) {
                    m_currentFilePos++;
                } else {
                    m_currentFilePos = 0;
                }
                PlayCurrentFile();

            }





        }
    }
    protected void PlayCurrentFile()
    {

        if (m_player != null) {//����Ѿ��ڲ����ˣ������´�������
            if (m_player.isPlaying())
            {
                m_player.stop();
            }
            m_player.release();
            m_player = new MediaPlayer();
        }
        else {
            m_player = new MediaPlayer();
        }
        m_player.setOnCompletionListener(new OnPlayCompletion());

        /*
        try {
            String s = m_AudioFileDir + "/" + m_titles[m_currentFilePos];
            m_player.setDataSource(MainActivity.this, Uri.parse(s));
            m_player.prepare();
            m_player.start();


        }
        catch (Exception e)
        {
            Toast.makeText(MainActivity.this, "io exception:"+e.toString(), 1000).show();
            return;
        }
         //�޸���ؽ���
        Button btn = (Button)(MainActivity.this.findViewById(R.id.btn_play));
        btn.setText("pause");

        TextView tv = (TextView)(MainActivity.this.findViewById(R.id.txtVw_ShowFileName));
        tv.setText(""+m_currentFilePos+" "+m_titles[m_currentFilePos]);

        //����һ���̣߳����½�����
        ProgressBar bar = (ProgressBar)(MainActivity.this.findViewById(R.id.progressBar));
        */

        findViewById(R.id.loadingPanel).setVisibility(View.VISIBLE);
        new Thread(new ThreadGetUrl(this ,m_handler)).start();







    }
    public void pauseOrContinue()
    {
        if (m_player != null)
        {
            View v = this.findViewById(R.id.btn_play);
            if (m_player.isPlaying())
            {
                m_player.pause();
                m_isPaused = true;
                ((Button)v).setText("continue");
            }
            else if (m_isPaused)
            {
                m_player.start();
                m_isPaused = false;
                ((Button)v).setText("  pause  ");
            }
        }
    }
    public void onButtonClicked(View v)
    {
        if (v.getId() == R.id.btn_play)
        {
            pauseOrContinue();
        }
        if (v.getId() == R.id.btn_back)
        {
            if (m_player != null) {
                if (m_player.isPlaying() || m_isPaused) {
                    List<Integer> brklist = m_brkpoints.get(m_titles[m_currentFilePos]);
                    int curPos = m_player.getCurrentPosition();
                    if (brklist == null||brklist.size() == 0) {
                        curPos = curPos - 5000;
                        if (curPos < 0) {
                            curPos = 0;
                        }
                    }
                    else
                    {
                        int i;
                        for (i = brklist.size()-1; i >= 0; --i)
                        {
                            if ((brklist.get(i).intValue()+500) < curPos)
                            {
                                curPos = brklist.get(i).intValue()+5;
                                m_lastbrk = brklist.get(i).intValue();

                                break;
                            }
                        }
                    }
                    m_player.seekTo(curPos);
                }
            }
        }
        if (v.getId() == R.id.btn_brk)
        {
            if (m_player != null) {
                if (m_player.isPlaying() ) {
                    int curPos = m_player.getCurrentPosition();
                    List<Integer> brklist = m_brkpoints.get(m_titles[m_currentFilePos]);
                    if (brklist == null) {
                        brklist = new ArrayList<>();
                        m_brkpoints.put(m_titles[m_currentFilePos], brklist);
                        Log.d("bison", "new break list for "+m_titles[m_currentFilePos]);
                    }
                    brklist.add(new Integer(curPos));
                    Collections.sort(brklist);
                    //Log.d("bison", "add break point at "+curPos+" for "+m_titles[m_currentFilePos]);
                }
            }

        }
        if (v.getId() == R.id.btn_clrbrk)
        {
            if (m_player != null) {
                if (m_player.isPlaying() || m_isPaused )
                {
                    new AlertDialog.Builder(this)
                            .setTitle("删除确认")
                            .setMessage("确定删除当前mp3的断句信息吗？")
                            .setPositiveButton("是", new DialogInterface.OnClickListener() {
                                public void onClick(DialogInterface dialog, int which) {
                                    MainActivity.this.m_brkpoints.put(m_titles[m_currentFilePos], new ArrayList<Integer>());
                                }

                            })
                            .setNegativeButton("否", null)
                            .show();


                }
            }

        }
        if (v.getId() == R.id.btn_forward)
        {
            if (m_player != null) {
                if (m_player.isPlaying() || m_isPaused) {
                    List<Integer> brklist = m_brkpoints.get(m_titles[m_currentFilePos]);
                    int curPos = m_player.getCurrentPosition();
                    if (brklist == null || brklist.size() == 0) {
                        int max = m_player.getDuration();
                        curPos = curPos + 5000;
                        if (curPos > max) {
                            curPos = max;
                        }
                    }
                    else
                    {
                        int i;
                        for (i = 0; i < brklist.size(); ++i)
                        {
                            if (brklist.get(i).intValue() > (curPos+500))
                            {
                                curPos = brklist.get(i).intValue()+5;
                                m_lastbrk = brklist.get(i).intValue();

                                break;
                            }
                        }
                    }
                    m_player.seekTo(curPos);
                }
            }
        }
        if (v.getId() == R.id.btn_next)
        {
            if (m_currentFilePos < (m_titles.length-1))
            {
                m_currentFilePos++;
            }
            else
            {
                m_currentFilePos = 0;
            }


            PlayCurrentFile();

        }
        if (v.getId() == R.id.btn_prev)
        {
            if (m_currentFilePos > 0)
            {
                m_currentFilePos--;
            }
            else
            {
                m_currentFilePos = m_titles.length-1;
            }



            PlayCurrentFile();

        }



    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d("bison","release media resource...");

        if (m_player != null) {
            if (m_player.isPlaying()) {
                m_player.stop();
            }
            m_player.release();
            m_player = null;
        }
        Log.d("bison", "saving break points...");

        SharedPreferences sp = this.getSharedPreferences("breakpoints", MODE_PRIVATE);
        SharedPreferences.Editor editor = sp.edit();
        for (String key:m_brkpoints.keySet())
        {
            List<Integer> values = m_brkpoints.get(key);
            Set<String> strSet = intList2StrSet(values);
            editor.putStringSet(key, strSet);
            Log.d("bison", "save string set for key:"+key+",size:"+strSet.size());

        }
        editor.commit();
    }
    public static Set<String> intList2StrSet(List<Integer> intlist)
    {
        Set<String> set = new HashSet<>();
        for (int i = 0; i < intlist.size(); i++) {
            set.add(intlist.get(i).toString());
        }
        return set;
    }
    public static List<Integer> strSet2IntList(Set<String> strset)
    {
        List<Integer> list = new ArrayList<>();
       Iterator<String> it = strset.iterator();
        while (it.hasNext())
        {
            list.add( new Integer(it.next())  );
        }
        Collections.sort(list);
        return list;
    }

    private void checkExternalStorage()
    {
        String status = Environment.getExternalStorageState();
        if (!status.equals(Environment.MEDIA_MOUNTED))
        {
            Toast.makeText(this, "no external storage.", 1000).show();
            return;
        }


        File sdcard = Environment.getExternalStorageDirectory();
        String dirStr = sdcard.getAbsolutePath()+"/"+"ListenEnglish/";
        File path = new File(dirStr);
        if (!path.exists())
        {
            if (!path.mkdir())
            {
                Toast.makeText(this, "mkdir "+path+" failed.", 1000).show();
                return;
            }

        }
        m_AudioFileDir = dirStr;
    }


}
