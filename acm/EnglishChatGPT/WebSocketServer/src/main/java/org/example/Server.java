package org.example;


import com.tencentcloudapi.common.profile.ClientProfile;
import com.tencentcloudapi.common.profile.HttpProfile;
import com.tencentcloudapi.tts.v20190823.TtsClient;
import com.tencentcloudapi.tts.v20190823.models.CreateTtsTaskRequest;
import com.tencentcloudapi.tts.v20190823.models.CreateTtsTaskResponse;
import com.tencentcloudapi.tts.v20190823.models.TextToVoiceRequest;
import com.tencentcloudapi.tts.v20190823.models.TextToVoiceResponse;
import org.java_websocket.WebSocket;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

import org.json.*;

import com.tencentcloudapi.common.Credential;
import com.tencentcloudapi.common.exception.TencentCloudSDKException;
import com.tencentcloudapi.cvm.v20170312.CvmClient;
import com.tencentcloudapi.cvm.v20170312.models.DescribeInstancesRequest;
import com.tencentcloudapi.cvm.v20170312.models.DescribeInstancesResponse;



public class Server extends WebSocketServer {

    static private String openai_api_key = System.getenv("OPENAI_API_KEY");

    public Server(int port) throws UnknownHostException {
        super(new InetSocketAddress(port));
    }

    public Server(InetSocketAddress address) {
        super(address);
    }

    @Override
    public void onMessage(WebSocket conn, String message) {

    }

    @Override
    public void onOpen(WebSocket conn, ClientHandshake handshake) {

        System.out.println(
                conn.getRemoteSocketAddress().getAddress().getHostAddress() + " entered the room!");

    }

    @Override
    public void onClose(WebSocket conn, int code, String reason, boolean remote) {
        broadcast(conn + " has left the room!");
        System.out.println(conn + " has left the room!");

    }

    @Override
    public void onMessage(WebSocket conn, ByteBuffer message) {
        super.onMessage(conn, message);
        int maxLen = 32*1024;
        byte[] buffer = new byte[maxLen];
        int i = 0;
        while (message.hasRemaining() && i < maxLen)
        {
            buffer[i++] = message.get();
        }

        System.out.println("receive message len: " + i);
        try {
            String filename = "/tmp/EnglishChatGPT_upstream.3gpp";
            String filename2 = "/tmp/EnglishChatGPT_upstream.m4a";
            FileOutputStream w = new FileOutputStream(filename);
            w.write(buffer, 0, i);
            w.close();
            transcode(filename, filename2);
            String text = audioTranscription(filename2, openai_api_key);
            if (text == null)
            {
                return;
            }
            System.out.println(text);
            List<String> response = chat(text, openai_api_key);
            if (response != null) {
                for (i = 0; i < response.size(); ++i) {
                    conn.send(response.get(i));
                    System.out.println("send message, length:"+response.get(i).length());
                }
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }
    public static List<String> splitLongText(String text)
    {
        String[] parts = text.split("[,.?;:]");

        int i = 0;
        for (i = 0; i < parts.length; ++i)
        {
            System.out.println(parts[i]);
        }
        System.out.println("\n\n");
        List<String> result = new ArrayList<String>();
        StringBuilder sb = new StringBuilder();
        i = 0;
        while (i < parts.length) {
            if (sb.length() < 140 && sb.length()+parts[i].length() < 140)
            {
                sb.append(parts[i]+",");
                i++;
            }
            else if (sb.length() > 0)
            {
                result.add(sb.toString());
                sb = new StringBuilder();
            }
            else // sb.length() == 0 and parts[i] is too long
            {
                result.add(parts[i]);
                i++;
            }
        }
        if (sb.length() > 0)
        {
            result.add(sb.toString());
        }
        for (i = 0; i < result.size(); ++i)
        {
            System.out.println(result.get(i));
        }
        System.out.println("\n");
        return result;
    }
    public static List<String> tts(String text) // 使用腾讯云TTS 服务
    {
        String secretID = System.getenv("TTS_SECRET_ID");
        String secretKey = System.getenv("TTS_SECRET_KEY");
        String filename = "/tmp/response.mp3";
        List<String> textList = splitLongText(text);
        List<String> audioList = new ArrayList<String>();
        int i;
        for (i = 0; i < textList.size(); ++i) {

            try {

                Credential cred = new Credential(secretID, secretKey);
                // 实例化一个http选项，可选的，没有特殊需求可以跳过
                HttpProfile httpProfile = new HttpProfile();
                httpProfile.setEndpoint("tts.tencentcloudapi.com");
                // 实例化一个client选项，可选的，没有特殊需求可以跳过
                ClientProfile clientProfile = new ClientProfile();
                clientProfile.setHttpProfile(httpProfile);
                // 实例化要请求产品的client对象,clientProfile是可选的
                TtsClient client = new TtsClient(cred, "ap-hongkong", clientProfile);
                // 实例化一个请求对象,每个接口都会对应一个request对象
                TextToVoiceRequest req = new TextToVoiceRequest();
                req.setText(textList.get(i));
                req.setSampleRate(8000L);
                req.setSessionId("" + (11228490+i));
                // 返回的resp是一个TextToVoiceResponse的实例，与请求对象对应
                TextToVoiceResponse resp = client.TextToVoice(req);
                System.out.println("tts:"+textList.get(i));
                JSONObject obj = new JSONObject(TextToVoiceResponse.toJsonString(resp));
                if (!obj.has("Audio")) {
                    return null;
                }
                String audioStr = obj.getString("Audio");
                audioList.add(audioStr);
                /*

                */


            } catch (Exception e) {
                System.out.println(e.toString());
                return null;
            }
        }
        return audioList;
    }
    public static List<String> chat(String in, String openai_api_key)//与openai交互，聊天
    {
        in = in.replace('\'', ' ');
        String cmd = String.join(" ", "curl https://api.openai.com/v1/chat/completions",
                "-H 'Content-Type: application/json'",
                "-H 'Authorization: Bearer "+openai_api_key+"'",
                "-d '{\"model\": \"gpt-3.5-turbo\", \"messages\": [{\"role\": \"user\", \"content\": \""+in+"\"}]",
                "}'");
        System.out.println(cmd);
        String cmds[] = {"sh", "-c", cmd};
        ProcessBuilder process = new ProcessBuilder(cmds );
        Process p;

        try {
            p = process.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
            StringBuilder builder = new StringBuilder();
            String line = null;
            while ((line = reader.readLine()) != null) {
                builder.append(line);
                builder.append(System.getProperty("line.separator"));
            }
            reader.close();
/*
            if (p.exitValue() != 0)
            {
                System.out.println("exit code:"+p.exitValue()+builder.toString());
                return null;
            }

 */
            System.out.println("response:"+builder.toString());
            if (builder.toString().length() < 2)
            {
                return null;
            }
            JSONObject obj = new JSONObject(builder.toString());
            if (obj.has("error"))
            {
                String errmsg = obj.getJSONObject("error").getString("message");
                System.out.println(builder.toString());
                return null;
            }
            else if (obj.has("choices"))
            {
                JSONArray array =  obj.getJSONArray("choices");
                String content = array.getJSONObject(0).getJSONObject("message").getString("content");
                System.out.println(content);
                List<String> audioList = tts(content);
                System.out.println("get response size:"+audioList.size());
                return audioList;
            }
            else
            {
                return null;
            }
        } catch (IOException e) {
            System.out.print("error");
            e.printStackTrace();
        }
        return null;
    }

    public static void transcode(String in, String out)//音频文件转码转封装
    {
        String cmds[] = {"ffmpeg", "-y",
                "-i", in,
                out};
        String longcmd = String.join(" ", cmds);
        System.out.println(longcmd);

        //其中-c表示cmd是一条命令，从而不会被截断.
        // 如果给ProcessBuilder直接传入cmd，会有意想不到的鸡巴错误
        String cmds2[] = {"sh", "-c", longcmd};
        ProcessBuilder process = new ProcessBuilder(cmds2 );
        Process p;

        try {
            p = process.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
            StringBuilder builder = new StringBuilder();
            String line = null;
            while ((line = reader.readLine()) != null) {
                builder.append(line);
                builder.append(System.getProperty("line.separator"));
            }
            reader.close();
        } catch (IOException e) {
            System.out.print("error");
            e.printStackTrace();
        }
    }
    public static String audioTranscription(String filename, String openai_api_key)//语音识别成位子
    {
        String cmds[] = {"curl", "https://api.openai.com/v1/audio/transcriptions",
                "-H", "'Authorization: Bearer " +openai_api_key +"'",
                "-H", "'Content-Type: multipart/form-data'",
                "-F", "file='@" + filename+"'",
                "-F", "model='whisper-1'"};
        String longcmd = String.join(" ", cmds);
        System.out.println(longcmd);


        for (String s : cmds)
        {
            System.out.println(s);
        }
        //其中-c表示cmd是一条命令，从而不会被截断.
        // 如果给ProcessBuilder直接传入cmd，会有意想不到的鸡巴错误
        String cmds2[] = {"sh", "-c", longcmd};
        ProcessBuilder process = new ProcessBuilder(cmds2 );
        Process p;

        try {
            p = process.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
            StringBuilder builder = new StringBuilder();
            String line = null;
            while ((line = reader.readLine()) != null) {
                builder.append(line);
                builder.append(System.getProperty("line.separator"));
            }
            reader.close();
/*
            if (p.exitValue() != 0)
            {
                System.out.println("exit code:"+p.exitValue()+builder.toString());
                return null;
            }

 */

            System.out.println("response:"+builder.toString());
            JSONObject obj = new JSONObject(builder.toString());
            if (obj.has("error"))
            {
                String errmsg = obj.getJSONObject("error").getString("message");
                System.out.println(builder.toString());
                return null;
            }
            else if (obj.has("text"))
            {
                return obj.getString("text");
            }
            else
            {
                return null;
            }
        } catch (IOException e) {
            System.out.print("error");
            e.printStackTrace();
        }
        return null;
    }



    @Override
    public void onError(WebSocket conn, Exception ex) {
        ex.printStackTrace();
        if (conn != null) {
            // some errors like port binding failed may not be assignable to a specific
            // websocket
        }

    }

    @Override
    public void onStart() {
        System.out.println("Server started!");
        //setConnectionLostTimeout(0);
        //setConnectionLostTimeout(100);


    }
    static public void main(String[] args)
    {
        //tts("This class consists exclusively of static methods for obtaining encoders and decoders for the Base64 encoding scheme. The implementation of this class supports the following types of Base64 as specified in RFC 4648 and RFC 2045.");

        try {
            Server s = new Server(7890);
            s.start();
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }



    }

}
