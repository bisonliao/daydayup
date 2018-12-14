package com.example.PCMPlayer;

import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.util.Log;

/**
 * Created by bisonliao on 2018/12/13.
 */
public class SimplePCMPlayer {
    private static final String TAG = "AudioOutputTrack";
    public static final int SAMPLES_PER_SECOND = 44100;
    public static final int BYTES_PER_SAMPLE = 2;
    public static final int BYTES_PRE_FRAME = SAMPLES_PER_SECOND * BYTES_PER_SAMPLE;
    private AudioTrack mAudioTrack;
    private int mSampleRate;
    private int minBufferSize;

    public SimplePCMPlayer() {

    }

    public void start(int sampleRate) {
        stop();
        Log.d("bison", "sampleRate:"+sampleRate);
        mSampleRate = sampleRate;
        mAudioTrack = createAudioTrack(mSampleRate);
        mAudioTrack.play();
    }

    public AudioTrack createAudioTrack(int sampleRate) {
        Log.d("bison", "sampleRate:"+sampleRate);
        int minBufferSizeBytes = AudioTrack.getMinBufferSize(sampleRate,
                AudioFormat.CHANNEL_OUT_STEREO,
                //AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT);
        Log.i(TAG, "AudioTrack.minBufferSize = " + minBufferSizeBytes
                + " bytes = " + (minBufferSizeBytes / BYTES_PRE_FRAME)
                + " frames");
        int bufferSize = 8*minBufferSizeBytes/8;
        minBufferSize = bufferSize;
        int outputBufferSizeFrames = bufferSize/BYTES_PRE_FRAME;
        Log.i(TAG, "actual bufferSize = " + bufferSize + " bytes = " + outputBufferSizeFrames + " frames");

        AudioTrack player = new AudioTrack(
                AudioManager.STREAM_MUSIC,//���������������ȼ��ģ�����绰�������Դ������
                mSampleRate,
                AudioFormat.CHANNEL_OUT_STEREO,AudioFormat.ENCODING_PCM_16BIT,
                bufferSize,AudioTrack.MODE_STREAM);

        Log.i(TAG, "created AudioTrack");


        return player;
    }


    public int write(byte[] buffer, int offset, int length) {
        return mAudioTrack.write(buffer,offset,length);
    }
    public int write(short[] buffer, int offset, int length){return mAudioTrack.write(buffer, offset, length);}

    public void stop() {
        if (mAudioTrack != null) {
            mAudioTrack.stop();
        }
    }

    public int getSampleRate() {
        return mSampleRate;
    }

    public AudioTrack getAudioTrack() {
        return mAudioTrack;
    }

    public int getMinBufferSize() {
        return minBufferSize;
    }
}
