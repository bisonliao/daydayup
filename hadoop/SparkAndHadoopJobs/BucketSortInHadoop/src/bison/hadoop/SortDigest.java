package bison.hadoop;

import java.io.IOException;
import java.lang.Iterable;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

/**
 * Created by Administrator on 2016/11/30.
 */
public class SortDigest {
    public static class DigestMapper extends Mapper<LongWritable, Text, Text, Text> {


        @Override
        public void map(LongWritable key, Text value, Mapper.Context context)
                throws IOException, InterruptedException {

            String valueStr = value.toString();
            ArrayList<String> strList = Tools.splitString(valueStr, ",");
            if (strList.size() != 2)
            {
                throw new IOException("invalid input:"+valueStr);
            }
            String md5str = strList.get(1);
            String bucket = Seperator100.getSeperator(md5str);
           // System.out.println("map "+valueStr+" to "+bucket);
            context.write(new Text(bucket), value );
        }
    }



    static class DigestReducer extends Reducer<Text, Text, Text, Text> {



        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {


            Map<String, String> strList = new TreeMap<String, String>(Seperator100.comp);


            for (Text v:values)
            {
                ArrayList<String> kv = Tools.splitString(v.toString(), ",");
                if (kv.size() != 2)
                {
                    throw  new IOException("invalid input:"+v.toString());
                }
                strList.put(kv.get(1), kv.get(0));
            }
            Iterator it = strList.keySet().iterator();
            while (it.hasNext())
            {
                String keyStr = (String)it.next();
                context.write(new Text(keyStr), new Text(strList.get(keyStr)));
            }




        }
    }

    public static void main(String[] args) throws Exception {
        /*
        * this example shows how to sort huge data with hadoop.
        *  The method is classical bucket sort algorithm.
        * It has to use customized partitioner，just like lots of sorted buckets
        * Each data falls into one bucket according its value,then we sort the data within each bucket.
        */
        //输入路径

        //String dst = "hdfs://namenode:9000//raw/raw_*";
        String dst = "/raw/raw_*";

        //输出路径，必须是不存在的，空文件加也不行。

        //String dstOut = "hdfs://namenode:9000/sorted2";
        String dstOut = "/Seperator100";
        //主类
        Configuration conf = new Configuration();

        final Job job = Job.getInstance(conf);
        job.setJarByClass(SortDigest.class);
// 寻找输入
        FileInputFormat.setInputPaths(job, dst);
// 1.2对输入数据进行格式化处理的类
        job.setInputFormatClass(TextInputFormat.class);
        job.setMapperClass(DigestMapper.class);

// 1.2指定map输出类型<key,value>类型
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

// 1.3指定分区
        //job.setPartitionerClass(HashPartitioner.class);
        job.setPartitionerClass(BisonPartitioner.class);
        job.setNumReduceTasks(Seperator100.getPartitionNumber());



// 1.4排序分组省略，使用默认
// 1.5规约省略，使用默认
        job.setReducerClass(DigestReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
// 指定输出路径
        FileOutputFormat.setOutputPath(job, new Path(dstOut));
// 指定输出的格式或则类
        job.setOutputFormatClass(TextOutputFormat.class);

// 把作业提交给jobtracer
        job.waitForCompletion(true);
    }

}
