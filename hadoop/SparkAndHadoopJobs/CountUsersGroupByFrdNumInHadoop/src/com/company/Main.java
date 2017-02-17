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
         * ������������ǣ����� <uin ������>������һ��һ�еļ�¼��ͳ�Ƴ� ����ĳ���������ľ�����û������������ <���Ѹ���  ���иú��и������û�����>
         *  ���룺uin ���Ѹ��� �����Ѹ����ķ�Χ[0, 1000]
         *  �����key 1������key�� ���Ѹ���-����±꣬����±귶Χ[0, 1000)�������Ǳ���ĳһ�������Ѹ�������uin̫�ർ��reduce�����е�����������
         *     ʹ�õ�����map���������ĳ�����������û����ر�࣬�����������ܴ����С
         */

        String dstOut = "hdfs://namenode:9000/output";
        //����
        Configuration conf = new Configuration();
        JobConf job = new JobConf(conf);
        job.setJobName("FriendNumber");
        job.setJarByClass(com.company.Main.class);

        job.setInputFormat(org.apache.hadoop.mapred.TextInputFormat.class);
        job.setOutputFormat(org.apache.hadoop.mapred.TextOutputFormat.class);
        // Ѱ������
        org.apache.hadoop.mapred.FileInputFormat.setInputPaths(job, new Path(dst));
        // ָ�����·��
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
