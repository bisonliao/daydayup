package ngse.org;

import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream;

import java.io.FileInputStream;
import java.io.FileOutputStream;

/**
 * Created by Administrator on 2016/4/7.
 */
public class GzipUtil {
    static public void zip(String srcFile) throws  Exception
    {
        GzipCompressorOutputStream out = new GzipCompressorOutputStream(new FileOutputStream(srcFile+".gz"));
        FileInputStream in = new FileInputStream(srcFile);
        byte[] buf = new byte[10240];
        while (true)
        {
            int len = in.read(buf);
            if (len <= 0)
            {
                break;
            }
            out.write(buf, 0, len);
        }
        out.flush();
        out.close();
        in.close();
    }
    static public void zip(String srcFile, String destFile) throws Exception
    {
        GzipCompressorOutputStream out = new GzipCompressorOutputStream(new FileOutputStream(destFile));
        FileInputStream in = new FileInputStream(srcFile);
        byte[] buf = new byte[10240];
        while (true)
        {
            int len = in.read(buf);
            if (len <= 0)
            {
                break;
            }
            out.write(buf, 0, len);
        }
        out.flush();
        out.close();
        in.close();
    }
    static public void unzip(String srcFile) throws Exception
    {
        GzipCompressorInputStream in = new GzipCompressorInputStream(new FileInputStream(srcFile));
        int index = srcFile.indexOf(".gz");
        String destFile = "";
        if (index == srcFile.length()-3)
        {
            destFile = srcFile.substring(0, index);
        }
        else
        {
            destFile = srcFile+".decompress";
        }
        FileOutputStream out = new FileOutputStream(destFile);
        byte[] buf = new byte[10240];
        while (true)
        {
            int len = in.read(buf);
            if (len <= 0)
            {
                break;
            }
            out.write(buf, 0, len);
        }
        out.flush();
        out.close();
        in.close();
    }
}
