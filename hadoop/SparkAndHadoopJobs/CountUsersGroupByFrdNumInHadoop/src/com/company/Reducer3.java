package com.company;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import java.io.IOException;
import java.util.Iterator;

/**
 * Created by Administrator on 2017/1/29.
 */
public class Reducer3 implements Reducer<LongWritable,LongWritable, LongWritable, LongWritable> {

    @Override
    public void reduce(LongWritable longWritable, Iterator<LongWritable> iterator, OutputCollector<LongWritable, LongWritable> outputCollector, Reporter reporter) throws IOException {
        long sum = 0;
        while (iterator.hasNext())
        {
            LongWritable v = iterator.next();
            sum += v.get();
        }
        outputCollector.collect(longWritable, new LongWritable(sum));
    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public void configure(JobConf jobConf) {

    }
}
