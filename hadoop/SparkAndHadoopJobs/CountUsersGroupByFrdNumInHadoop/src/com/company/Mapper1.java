package com.company;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;


import java.io.IOException;
import java.security.SecureRandom;

/**
 * Created by Administrator on 2017/1/29.
 */
public  class Mapper1 implements Mapper<LongWritable, Text, Text, Text> {
    static private SecureRandom random = null;
    static
    {
        try {
            random = SecureRandom.getInstance("SHA1PRNG");
            random.setSeed((long)(Math.random() * 1000000));
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

    }

    @Override
    public void map(LongWritable longWritable, Text text, OutputCollector<Text, Text> outputCollector, Reporter reporter) throws IOException {
        //���룺uin ���Ѹ��� �����Ѹ����ķ�Χ[0, 1000]
        // �����key 1������key�� ���Ѹ���-����±꣬����±귶Χ[0, 1000)�������Ǳ���ĳһ�������Ѹ�������uin̫�ർ��reduce�����е�����������

        String line = text.toString();
        int sepIndex = line.indexOf(" ");
        if (sepIndex > 0)
        {
            String uinStr = line.substring(0, sepIndex);
            String frdNumStr = line.substring(sepIndex + 1);

            long frdNum = new Long(frdNumStr).longValue();
            if (frdNum < 0 || frdNum > 1000)
            {
                return;
            }

            int suffix = random.nextInt();
            if (suffix < 0) { suffix = - suffix;}
            suffix = suffix % 1000;
            String mapKey = ""+frdNum+"-"+suffix;
            outputCollector.collect(new Text(mapKey),new Text("1"));


        }
    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public void configure(JobConf jobConf) {

    }


}
