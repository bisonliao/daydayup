package ngse.org;

import beans.request.GetVersionListRequest;
import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.HttpStatus;
import org.apache.commons.httpclient.methods.GetMethod;
import org.jfree.chart.*;

import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.axis.DateTickUnit;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.time.Hour;
import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;
import org.jfree.data.xy.XYDataset;
import org.json.JSONObject;

import javax.imageio.ImageIO;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.charset.Charset;
import java.security.MessageDigest;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by Administrator on 2016/2/10.
 */
public class Tools {

    static public  String randInt()
    {
        return String.format("%d", (int)(Math.random()*Integer.MAX_VALUE));
    }

    //长度整数转化为大端四字节整数
    static public byte[] int2Bytes(int i)
    {
        byte[] b = new byte[4];
        int v = 256 * 256 * 256;
        for (int j = 0; j < 3; j++) {
            b[j] = (byte)(i / v);
            i = i % v;
            v = v / 256;
        }
        b[3] = (byte)i;

        return b;
    }
    static public int bytes2int(byte[] buf)
    {
        int v = 0;
        int b0 = buf[0]; if (b0 < 0) { b0 += 256;}
        int b1 = buf[1]; if (b1 < 0) { b1 += 256;}
        int b2 = buf[2]; if (b2 < 0) { b2 += 256;}
        int b3 = buf[3]; if (b3 < 0) { b3 += 256;}
        v = b0 * (256*256*256) + b1 * (256*256) + b2*256 + b3;
        return v;
    }


    //分号分割的字符串分割为字符串列表
    public static ArrayList<String> splitBySemicolon(String s)
    {
        ArrayList<String> ret = splitString(s, ";");
        return ret;
    }
    //分号分割的字符串分割为字符串列表
    public static ArrayList<String> splitString(String s, String sep)
    {
        ArrayList<String> ret = new ArrayList<String>();
        int fromIndex = 0;
        while (true)
        {
            int index = s.indexOf(sep, fromIndex);
            if (index >= 0)
            {
                String sub = s.substring(fromIndex, index);
                if (sub.length() > 0)
                {
                    ret.add(sub);
                }
                fromIndex = index + 1;
            }
            else
            {
                if (fromIndex < s.length())
                {
                    String sub = s.substring(fromIndex);
                    if (sub.length() > 0)
                    {
                        ret.add(sub);
                    }
                }
                break;
            }
        }
        return ret;
    }


    public static HashMap<String, String> checkAccessToken(String access_token) throws IOException,NeedLoginException
    {


        String jsonStr = "";

        HttpClient client = new HttpClient();
        client.getHttpConnectionManager().getParams().setConnectionTimeout(3000);
        client.getHttpConnectionManager().getParams().setSoTimeout(3000);
        GetMethod method = new GetMethod("https://graph.qq.com/oauth2.0/me?access_token=" + access_token);

        int status = client.executeMethod(null, method);
        if (status != HttpStatus.SC_OK) {
            throw new IOException("call graph.qq.com failed");
        }

        byte[] body = method.getResponseBody();
        jsonStr = new String(body, Charset.forName("utf8"));
        Pattern p = Pattern.compile("\\{.*\\}");
        Matcher m = p.matcher(jsonStr);
        if (!m.find()) {
            throw new IOException("graph.qq.com returns invalid json string.");
        }
        jsonStr = m.group(0);


        JSONObject object = new JSONObject(jsonStr);
        if (object == null || !object.has("client_id") || !object.has("openid"))
        {
            throw new NeedLoginException("invalid access token.graph return:"+jsonStr);
        }
        String client_id = object.getString("client_id");
        String open_id = object.getString("openid");
        if (client_id == null || client_id.length()<1)//没有返回有效的身份
        {
            throw new NeedLoginException("invalid access token");
        }

        HashMap<String, String> map = new HashMap<>();
        map.put("client_id", client_id);
        map.put("openid", open_id);

        return map ;
    }


    static public int[] zeroIntArray(int size)
    {
        int [] ret = new int[size];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = 0;
        }
        return  ret;
    }


        //fork进程执行命令
    // cmd: 命令文件
    // sb： 标准输出和标准错误输出的内容保存，可以为null
    // waitFlag：是否等子进程结束再函数返回
    static public int runCommand(String[] cmd, StringBuffer sb, boolean waitflag )
    {

        Process pid = null;
        ProcessBuilder build = new ProcessBuilder(cmd);
        build.redirectErrorStream(true);
        try {
            pid = build.start();
        }
        catch (Exception e)
        {
            e.printStackTrace();
            return -1;
        }
        if (sb != null) {
            //BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(pid.getInputStream()), 1024);
            InputStream in = pid.getInputStream();
            byte[] buf = new byte[10240];
            try {
                while (true)
                {
                    int len = in.read(buf);
                    if (len <= 0)
                    {
                        break;
                    }
                    sb.append(new String(buf, 0, len));
                }
            }
            catch (Exception e)   { }

        }
        if (waitflag) {
            try {
                pid.waitFor();
                int v = pid.exitValue();
                pid.destroy();
                return v;
            }catch (Exception e ){}
        }
        return 0;
    }
    //删除目录，目录可以为非空，递归的方式删除子项
    static public boolean deleteDirectory(File path)
    {

        if( path.exists() ) {
            File[] files = path.listFiles();
            for(int i=0; i<files.length; i++) {
                if(files[i].isDirectory()) {
                    deleteDirectory(files[i]);
                }
                else {
                    files[i].delete();
                }
            }
        }
        return( path.delete() );

    }

    static public String md5(String s)
    {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");

            md.update(s.getBytes());
            byte[] result = md.digest();
            if (result.length != 16)
            {
                return "";
            }

            return Tools.toHexString(result);// 32 bytes



        }
        catch (Exception e )
        {
            return "";
        }
    }
    static public String toHexString(byte[] b)
    {
        int i;
        StringBuffer sb = new StringBuffer();
        char[] chars = {'0', '1','2','3', '4','5','6','7','8','9','a','b','c','d','e','f'};
        for (i = 0; i < b.length; ++i)
        {
            int bb = b[i];
            if (bb < 0) { bb += 256;}
            int index;
            index = bb>>4;
            sb.append(chars[index]);
            index = bb & 0x0f;
            sb.append(chars[index]);
        }
        return sb.toString();
    }
    static private  int hexChr2Int(char c)
    {
        char[] chars = {'0', '1','2','3', '4','5','6','7','8','9','a','b','c','d','e','f'};
        int i;
        for (i = 0; i < chars.length; ++i)
        {
            if (chars[i] == c)
            {
                return i;
            }
        }
        return 16;
    }
    static public byte[] fromHexString(String s)
    {
        int i;
        if ((s.length() % 2) != 0)
        {
            return new byte[0];
        }
        int len = s.length() / 2;
        byte[] b = new byte[len];


        for (i = 0; i < b.length; ++i)
        {
            int v1 = hexChr2Int(s.charAt(2*i));
            int v2 = hexChr2Int(s.charAt(2*i+1));
            if (v1 > 15 || v2 > 15) { return new byte[0];}
            b[i] = (byte)(v1*16+v2);
        }
        return b;
    }
    static public String nowString(String fmt)
    {
        // "yyyy-MM-dd HH:mm:ss"
        if (fmt == null || fmt.length() < 1)
        {
            fmt = "yyyy-MM-dd HH:mm:ss";
        }
        SimpleDateFormat df = new SimpleDateFormat(fmt); //设置日期格式
        return df.format(new Date());

    }
    static public String TimeStamp2DateStr(Long epochSecond, String formats){
        String date = new java.text.SimpleDateFormat(formats).format(new java.util.Date(epochSecond * 1000));
        return date;
    }
    static public String TimeStamp2DateStr(Long epochSecond)
    {
        return TimeStamp2DateStr(epochSecond,  "yyyy-MM-dd HH:mm:ss");
    }
}
