package bison.hive;

import java.nio.charset.Charset;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by Administrator on 2016/11/24.
 */
public class GenRandLog {

    private static String keychars = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
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

    public Map<String, String> getRandLogString() throws Exception
    {

        int max = keychars.length();
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < 10; ++i) {
            int v = r.nextInt(max);
            sb.append(keychars.substring(v, v+1));
        }
        Map<String, String> maps = new HashMap<String, String>();
        String requestID = sb.toString();
        maps.put("RequestID", requestID);

        MessageDigest digest = MessageDigest.getInstance("MD5");
        digest.update(requestID.getBytes(Charset.forName("utf8")));
        byte[] md5 = digest.digest();
        String md5str = toHexString(md5);

        maps.put("MD5", md5str);

        return  maps;
    }

}
