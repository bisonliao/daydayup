package com.example.PCMPlayer;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;

/**
 * Created by bisonliao on 2018/12/14.
 */
public class SimplePCMRecorder {
    private AudioRecord mRecorder = null;

    public SimplePCMRecorder()
    {

    }
    public void start(int sampleRate)
    {
        for (int i = 0; i < 3; ++i) {
            int buffSize = AudioRecord.getMinBufferSize(sampleRate,
                    AudioFormat.CHANNEL_IN_STEREO,
                    AudioFormat.ENCODING_PCM_16BIT);//计算最小缓冲区
            mRecorder = new AudioRecord(MediaRecorder.AudioSource.MIC,
                    sampleRate,
                    AudioFormat.CHANNEL_IN_STEREO,
                    AudioFormat.ENCODING_PCM_16BIT, buffSize);//创建AudioRecorder对象

            if (mRecorder.getState() == AudioRecord.STATE_INITIALIZED) {
                break;
            }
        }
        mRecorder.startRecording();
    }

    public int read(short[] buff,  int length)
    {
        if (mRecorder == null) { return -1;}
        int offset = 0;
        while (offset < length)
        {
            if (mRecorder.getRecordingState() != AudioRecord.RECORDSTATE_RECORDING)
            {
                break;
            }
            int len = mRecorder.read(buff, offset, length - offset);
            if (len > 0)
            {
                offset += len;
            }
            else
            {
                break;
            }
        }
        return offset;
    }
    public  void stop()
    {
        if (mRecorder!=null) {
            mRecorder.stop();
            mRecorder.release();
            mRecorder = null;
        }
    }

}
