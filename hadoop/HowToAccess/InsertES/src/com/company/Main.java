package com.company;

import java.net.SocketTimeoutException;
import java.util.concurrent.ExecutionException;

public class Main {

    static public void main(String[] args) {
        String url = "http://datanode2:9200/msec_log/business/_bulk";
        if (args.length >= 1)
        {
            url = args[0];
            System.out.println("url="+url);
        }
        int index = 0;
        while (true) {

            StringBuffer requestBody = new StringBuffer();
            for (int i = 0; i < 1000; ++i) {
                String now = Tools.nowString("yyyyMMddHHmmssSSS");
                ++index;
                String svcname = "voa.crawl" + (index % 30);

                String logstr = "{\"tt\":\"" + now +
                        "\", \"svcname\":\"" + svcname +
                        "\",\"loglevel\":\"INFO\", \"content\":\"hello world, what is you name? My name is hanmemmei. I, Grass\"}\n";
               // System.out.println(logstr);
                String action = "{ \"index\": {}}\n";
/*
                byte[] intBytes = Tools.fromHexString(Tools.md5(logstr).substring(0, 8)); // get first 4 bytes of md5
                int id = Math.abs(Tools.bytes2int(intBytes));
                */
                requestBody.append(action);
                requestBody.append(logstr);

            }

            StringBuffer response = new StringBuffer();
            try {
                AccessES.DoHttpPost(url, null,requestBody.toString(), response);
                System.out.println(Tools.getCurrentTime()+":"+response.substring(0, 200));
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }



        }
    }
}
