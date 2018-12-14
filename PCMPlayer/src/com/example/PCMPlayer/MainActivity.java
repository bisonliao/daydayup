package com.example.PCMPlayer;

import android.app.Activity;
import android.app.AlertDialog;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.widget.Button;

public class MainActivity extends Activity {

    private SimplePCMPlayer mPlayer = new SimplePCMPlayer();
    private MsgHandler mHandler = new MsgHandler();
    private SimplePCMRecorder mRecorder = new SimplePCMRecorder();
    private short[] recordedPCM = new short[SAMPLE_NUM];
    private int recordedLen = 0;
    public static final int SAMPLE_NUM = 44100*10;
    protected  boolean isRecording = false;
    protected  boolean isPlaying = false;
    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
    }

    public void onBtnClick(View v)
    {
        int id = v.getId();
        if (id == R.id.btn_play)
        {
            if (recordedLen < 10)
            {
                new AlertDialog.Builder(this)
                        .setTitle("提醒")
                        .setMessage("没有PCM数据，请先录制！")
                                .setPositiveButton("OK",null).show();
                return;
            }
            if (isPlaying)
            {
                return;
            }
            mPlayer.start(44100);
            new Thread(new ThreadWritePCM(mPlayer , mHandler,  recordedPCM, recordedLen)).start();
            ((Button)v).setText("正在播放...");
            isPlaying = true;
        }
        if (id == R.id.btn_record)
        {
            if (isRecording)
            {
                return;
            }
            mRecorder.start(44100);
            new Thread(new ThreadReadPCM(mRecorder, mHandler, recordedPCM, MainActivity.SAMPLE_NUM)).start();
            ((Button)v).setText("正在录音...");
        }

    }
    private class MsgHandler extends Handler
    {
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);

            if (msg.what == R.integer.WR_OVER)
            {
                Log.d("bison", "PCM data all played, stop the player...");
                mPlayer.stop();
                MainActivity.this.isPlaying =false;
                ((Button)MainActivity.this.findViewById(R.id.btn_play)).setText("播放");

            }
            if (msg.what == 23)
            {

                Log.d("bison", "PCM recording done!,datalen=" + msg.arg1 + " stop the recorder... ");
                mRecorder.stop();
                recordedLen = msg.arg1;
                ((Button)MainActivity.this.findViewById(R.id.btn_record)).setText("录音10s");
                MainActivity.this.isRecording =false;

                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("录音完成")
                        .setMessage("录得音频时长："+recordedLen/44100.0+"秒")
                        .setPositiveButton("OK",null).show();
            }
        }
    }
}
