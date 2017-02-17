package com.company;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.lib.ChainMapper;


import org.apache.hadoop.mapred.lib.ChainReducer;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

import java.util.Iterator;

public class Main {

    public static void main(String[] args) throws Exception{
        String dst = "hdfs://namenode:9000/frdNum.txt";

       /*
         * 下面这个例子是：输入 <uin 好友数>这样的一行一行的记录，统计出 具有某个好友数的具体的用户数量，即输出 <好友个数  具有该好有个数的用户数量>
         *  输入：uin 好友个数 ，好友个数的范围[0, 1000]
         *  输出：key 1，其中key是 好友个数-随机下标，随机下标范围[0, 1000)，作用是避免某一个“好友个数”的uin太多导致reduce过程中单机处理不过来
         *     使用到两次map，避免具有某个好友数的用户量特别多，超出单机所能处理大小
         */

        String dstOut = "hdfs://namenode:9000/output";
        //主类
        Configuration conf = new Configuration();
        JobConf job = new JobConf(conf);
        job.setJobName("FriendNumber");
        job.setJarByClass(com.company.Main.class);

        job.setInputFormat(org.apache.hadoop.mapred.TextInputFormat.class);
        job.setOutputFormat(org.apache.hadoop.mapred.TextOutputFormat.class);
        // 寻找输入
        org.apache.hadoop.mapred.FileInputFormat.setInputPaths(job, new Path(dst));
        // 指定输出路径
        org.apache.hadoop.mapred.FileOutputFormat.setOutputPath(job, new Path(dstOut));


        ChainMapper.addMapper(job, com.company.Mapper1.class,
                LongWritable.class,
                Text.class,
                Text.class,
                Text.class,
                true, new JobConf(false));
        ChainMapper.addMapper(job, com.company.Mapper2.class,
                Text.class,
                Text.class,
                LongWritable.class,
                LongWritable.class,
                true, new JobConf(false));

        ChainReducer.setReducer(job, com.company.Reducer3.class,
                LongWritable.class,
                LongWritable.class,
                LongWritable.class,
                LongWritable.class,
                true, new JobConf(false));
        JobClient.runJob(job);



    }
}
