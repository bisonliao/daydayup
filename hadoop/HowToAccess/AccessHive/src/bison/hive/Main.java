package bison.hive;

import javax.print.attribute.standard.RequestingUserName;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Map;

public class Main {

    public static void main(String[] args) throws Exception
    {

        GenRandLog log = new GenRandLog();

        for (int k = 10; k < 50; ++k) {

            OutputStream out = new FileOutputStream("/data/data0/log."+k);

            for (int i = 0; i < 100000; i++) {
                Map<String, String> maps = log.getRandLogString();
                String requestID = maps.get("RequestID");
                String md5str = maps.get("MD5");
                for (int j = 0; j < 10; j++) {
                    String content = "content#" + j + " of " + requestID;

                    // String sql = "insert into table log values('%s','%s','%s')";
                    String sql = "%s\001%s\001%s\001\n";
                    sql = String.format(sql, requestID, md5str, content);
                    out.write(sql.getBytes());
                    //System.out.println(sql);
                    //util.insert(sql);
                }

            }
            out.close();
        }

    }
}
