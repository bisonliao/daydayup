package bison.hadoop;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Partitioner;

/**
 * Created by Administrator on 2016/12/1.
 */
public class BisonPartitioner extends Partitioner<Text, Text> {
    @Override
    public int getPartition(Text s, Text s2, int i)
    {
        if (i != Seperator100.getPartitionNumber())
        {
            System.err.println(">>>>>>partition number invalid:" + i);
            return 0;
        }
        int index = Seperator100.getPartitionIndex(s.toString());
        if (index < 0)
        {
            System.err.println(">>>>>> can NOT find partition for key:"+s);
            return 0;
        }
        System.out.println("partition number of "+s.toString()+" is "+index);
        return index;
    }
}
