package com.company;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import java.io.IOException;
import java.util.Iterator;

/**
 * Created by Administrator on 2017/1/29.
 */
public  class Mapper2 implements Mapper<Text, Text, LongWritable, LongWritable> {

    long getTokenCount(String s) {
        if (s.length() < 1) {
            return 0;
        }
        long result = 1;
        while (true) {
            int sepIndex = s.indexOf(" ");
            if (sepIndex <= 0 || sepIndex >= (s.length() - 1)) {
                break;
            }
            s = s.substring(sepIndex + 1);
            result++;
        }
        return result;
    }

    @Override
    public void map(Text kk, Text vv, OutputCollector<LongWritable, LongWritable> outputCollector, Reporter reporter) throws IOException {
        //其中keyStr是 好友个数-随机下标，随机下标范围[0, 1000)
        // valueStr是数字1的列表

        String keyStr = kk.toString();
        String valueStr = vv.toString();


        int sepIndex = keyStr.indexOf("-");
        if (sepIndex <= 0)
        {
            return;
        }
        String frdNumStr = keyStr.substring(0, sepIndex);
        long frdNum = new Long(frdNumStr).longValue();

        long count = getTokenCount(valueStr);
        outputCollector.collect(new LongWritable(frdNum), new LongWritable(count));
    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public void configure(JobConf jobConf) {

    }
}
