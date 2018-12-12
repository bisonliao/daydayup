package ngse.org;

import org.json.JSONObject;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;

/**
 * Created by Administrator on 2016/3/9.
 */
public class RemoteShell {

    private String serverIP = "127.0.0.1";
    private int serverPort = 9981;

    public RemoteShell()
    {

    }
     private String getLengthField(int len)
    {
        StringBuffer sb = new StringBuffer();
        sb.append(new Integer(len).toString());
        while (sb.length() < 10)
        {
            sb.append(" ");
        }
        return sb.toString();
    }
    public  String SendFileToAgent(String localFileFullName, String remoteFileFullName, String remoteServerIP)
    {
        Socket socket = new Socket();

        try {
            socket.setSoTimeout(60000);
            socket.connect(new InetSocketAddress(serverIP,serverPort), 5000);

            OutputStream out = socket.getOutputStream();
            InputStream in = socket.getInputStream();



            String request = "{\"handleClass\":\"comm_with_client.service.SendFileToAgent\", \"requestBody\":{"+
                    "\"localFileFullName\":\""+localFileFullName+
                    "\",\"remoteFileFullName\":\""+remoteFileFullName+
                    "\",\"remoteServerIP\":\""+remoteServerIP+
                    "\"}}";
           // request = getLengthField(request.length())+request;

           // System.out.printf("send:%s\n", request);
            out.write(request.getBytes());
            socket.shutdownOutput();

            //读，一直读到对端关闭写
            byte[] buf = new byte[10240];
            int offset = 0;
            while (true)
            {
                if (offset >= buf.length)
                {
                    return "buffer size is too small";
                }
                int len = in.read(buf, offset, buf.length-offset);
                if (len < 0)
                {
                    break;
                }
                offset += len;
            }
            String jsonStr = new String(buf, 0, offset);

            JSONObject obj = new JSONObject(jsonStr);
            int status = obj.getInt("status");
            if (status == 0)
            {
                return "success";
            }
            else
            {
                return obj.getString("message");
            }

        }
        catch (Exception e)
        {
            e.printStackTrace();
            return e.getMessage();
        }
        finally {
            try {socket.close();}catch (Exception e){}
        }
    }
    public  String GetFileFromAgent(String localFileFullName,
                                            String remoteFileFullName,
                                            String remoteServerIP)
    {
        Socket socket = new Socket();

        try {
            socket.setSoTimeout(60000);
            socket.connect(new InetSocketAddress(serverIP,serverPort), 5000);

            OutputStream out = socket.getOutputStream();
            InputStream in = socket.getInputStream();




            String request = "{\"handleClass\":\"comm_with_client.service.GetFileFromAgent\", \"requestBody\":{\"localFileFullName\":\""+localFileFullName+
                    "\",\"remoteFileFullName\":\""+remoteFileFullName+
                    "\",\"remoteServerIP\":\""+remoteServerIP+
                    "\"}}";
           // request = getLengthField(request.length())+request;
           // System.out.printf("send:%s\n", request);
            out.write(request.getBytes());
            socket.shutdownOutput();

            //读，一直读到对端关闭写
            byte[] buf = new byte[10240];
            int offset = 0;
            while (true)
            {
                if (offset >= buf.length)
                {
                    return "buffer size is too small";
                }
                int len = in.read(buf, offset, buf.length-offset);
                if (len < 0)
                {
                    break;
                }
                offset += len;
            }
            String jsonStr = new String(buf, 0, offset);
            //检查结果
            JSONObject obj = new JSONObject(jsonStr);
            int status = obj.getInt("status");
            if (status == 0)
            {
                return "success";
            }
            else
            {
                return obj.getString("message");
            }

        }
        catch (Exception e)
        {
            e.printStackTrace();
            return e.getMessage();
        }
        finally {
            try {socket.close();}catch (Exception e){}
        }
    }
    public  String SendCmdsToAgentAndRun(String localFileFullName, String remoteServerIP, StringBuffer outputFileName )
    {
        Socket socket = new Socket();

        try {
            socket.setSoTimeout(60000);
            socket.connect(new InetSocketAddress(serverIP,serverPort), 5000);

            OutputStream out = socket.getOutputStream();
            InputStream in = socket.getInputStream();




            String request = "{\"handleClass\":\"comm_with_client.service.SendCmdsToAgentAndRun\", \"requestBody\":{\"localFileFullName\":\""+localFileFullName+
                    "\",\"remoteServerIP\":\""+remoteServerIP+
                    "\"}}";
           // request = getLengthField(request.length())+request;
           // System.out.printf("send:%s\n", request);
            out.write(request.getBytes());
            socket.shutdownOutput();

            //读，一直读到对端关闭写
            byte[] buf = new byte[10240];
            int offset = 0;
            while (true)
            {
                if (offset >= buf.length)
                {
                    return "buffer size is too small";
                }
                int len = in.read(buf, offset, buf.length-offset);
                if (len < 0)
                {
                    break;
                }
                offset += len;
            }
            String jsonStr = new String(buf, 0, offset);
            //检查结果
            JSONObject obj = new JSONObject(jsonStr);
            int status = obj.getInt("status");
            if (status == 0)
            {
                outputFileName.append(obj.getString("outputFileName"));
                return "success";
            }
            else
            {
                return obj.getString("message");
            }


        }
        catch (Exception e)
        {
            e.printStackTrace();
            return e.getMessage();
        }
        finally {
            try {socket.close();}catch (Exception e){}
        }
    }


}
