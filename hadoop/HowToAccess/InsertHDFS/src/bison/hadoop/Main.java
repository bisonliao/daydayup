package bison.hadoop;

import java.nio.charset.Charset;

public class Main {

    public static void main(String[] args)  throws Exception
    {


        RandPasswd rp = new RandPasswd();
        for (int i = 0; i < 300; i++) {
            String filename = rp.getRandFilename();
            filename = "hdfs://namenode:9000/raw/"+filename;
            HdfsHandler handler = HdfsHandler.openFileForWrite(filename, 1);
            for (int j = 0; j < 1000000; j++) {
                String content = rp.getRandPasswdAndMD5();
                byte[] buf = content.getBytes(Charset.forName("utf8"));
                handler.writeBytes(buf, 0, buf.length);
            }
            handler.close();
        }


    }
}
