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
        //输入：uin 好友个数 ，好友个数的范围[0, 1000]
        // 输出：key 1，其中key是 好友个数-随机下标，随机下标范围[0, 1000)，作用是避免某一个“好友个数”的uin太多导致reduce过程中单机处理不过来

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
