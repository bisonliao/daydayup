package com.example.englishchatgpt;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.media.AsyncPlayer;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.view.MotionEvent;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Deque;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    String mExternalStoragePath = Environment.getExternalStorageDirectory().getPath().toString();
    MediaRecorder recorder = null;
    Button   btnSpeak = null;
    ListView responseList = null;
    String  filepathSaveISpeak = null;



    private Deque<String> mAudioFileToServer;
    private List<String> mAudioFileFromServer = new ArrayList<String>();

    private int currentPlayIndex = 0;

    protected void playResponse(int index)
    {
        if (index+1 <= mAudioFileFromServer.size())
        {
            String filename = MainActivity.this.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath() + "response.mp3";

            byte[] audio = Base64.getDecoder().decode(mAudioFileFromServer.get(index));
            try {
                FileOutputStream out = new FileOutputStream(filename);
                out.write(audio);
                out.close();
            }catch (Exception e)
            {
                e.printStackTrace();
                return;
            }


            MediaPlayer player = MediaPlayer.create(getApplicationContext(), Uri.fromFile(new File(filename)));
            player.start();
            player.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
                    @Override
                    public void onCompletion(MediaPlayer mediaPlayer) {
                        mediaPlayer.release();

                    }
            });
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        String [] permissions =
        {
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.READ_MEDIA_AUDIO,
                Manifest.permission.INTERNET
        };
        ActivityCompat.requestPermissions(this, permissions, 200);

        filepathSaveISpeak = MainActivity.this.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath() + "ISpeak.3gpp";
        mAudioFileToServer = new ArrayDeque<String>();
        mAudioFileFromServer = new ArrayList<String>();
        responseList = (ListView)MainActivity.this.findViewById(R.id.ResponeList);


        Handler handler = new Handler()
        {
            @Override
            public void handleMessage(@NonNull Message msg) {
                super.handleMessage(msg);
                switch (msg.what)
                {
                    case 0:
                        List<String> items = new ArrayList<>();
                        int i;
                        for (i = 0; i < mAudioFileFromServer.size(); ++i)
                        {
                            items.add("voice#"+i);
                        }
                        ArrayAdapter<String> arrayAdapter = new ArrayAdapter<String>(MainActivity.this,
                                        android.R.layout.simple_expandable_list_item_1,
                                items);
                        responseList.setAdapter(arrayAdapter);
                        break;

                }
            }
        };

        new Thread(new FileTransfer(mAudioFileToServer,  mAudioFileFromServer, handler)).start();

        btnSpeak = (Button) this.findViewById(R.id.ISpeakBtn);
        btnSpeak.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                if (motionEvent.getAction() == MotionEvent.ACTION_DOWN)
                {
                    btnSpeak.setText("recording...");
                    onSomethingClicked(view);
                }
                else if (motionEvent.getAction() == MotionEvent.ACTION_UP)
                {
                    recorder.stop();
                    recorder.release();

                    btnSpeak.setText("push me to speak");

                    File f = new File(filepathSaveISpeak);
                    String s = "file length: "+f.length();
                    Toast.makeText(MainActivity.this, s, Toast.LENGTH_SHORT).show();

                    synchronized (mAudioFileToServer)
                    {
                        mAudioFileToServer.push(filepathSaveISpeak);
                    }

                }
                return false;
            }
        });
        responseList.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
               // Toast.makeText(MainActivity.this, "selected:"+i+" "+l, 1000).show();
                ((TextView)findViewById(R.id.voiceIndexShow)).setText("voice index:"+i);
                playResponse(i);
            }
        });


    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }



    public void onSomethingClicked(View v)
    {
        if (v.getId() == R.id.ISpeakBtn)
        {
            try {
                File f = new File(filepathSaveISpeak);
                if (f.exists())
                {
                    f.delete();
                }


                recorder  = new MediaRecorder();
                recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
                recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
                recorder.setOutputFile(filepathSaveISpeak);

                recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);

                recorder.prepare();
                recorder.start();

            }
            catch (Exception e)
            {
                e.printStackTrace();
                Toast.makeText(this, e.getMessage()+e.toString(), 1000).show();
                return;
            }

        }
    }
}