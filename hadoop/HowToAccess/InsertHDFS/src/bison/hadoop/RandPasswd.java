package bison.hadoop;

import java.nio.charset.Charset;
import java.security.MessageDigest;
import java.security.SecureRandom;

/**
 * Created by Administrator on 2016/11/23.
 */
public class RandPasswd {

    private static String passwdchars = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    SecureRandom r = new SecureRandom();
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

    public  String getRandPasswdAndMD5() throws Exception
    {

        int max = passwdchars.length();
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < 10; ++i) {
            int v = r.nextInt(max);
            sb.append(passwdchars.substring(v, v+1));
        }
        String passwd = sb.toString();
        MessageDigest digest = MessageDigest.getInstance("MD5");
        digest.update(passwd.getBytes(Charset.forName("utf8")));
        byte[] md5 = digest.digest();
        String md5str = toHexString(md5);
        return passwd+","+md5str+"\n";

    }
    public String getRandFilename() throws Exception
    {
        int i;
        StringBuffer sb = new StringBuffer();
        sb.append("raw_");
        char[] chars = {'0', '1','2','3', '4','5','6','7','8','9','a','b','c','d','e','f'};
        for (i = 0; i < 10; ++i)
        {
            int index = r.nextInt(chars.length);
            sb.append(chars[index]);
        }
        return sb.toString();
    }
}
