package beans.service;


import beans.request.JoinGroupRequest;
import ngse.org.*;
import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.HttpStatus;
import org.apache.commons.httpclient.cookie.CookiePolicy;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.log4j.Logger;


import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


/**
 * Created by Administrator on 2016/10/26.
 */
public class JoinGroup extends JsonRPCHandler {


    public static void main(String args[])
    {
        joinGroup("11228490", "%40pv45XRjBS", "9E0FEBB904A87131897428C21BC45B9A", "119.27.47.9", "22023456", "8132D5F80AA27F67A7C5F78AFC0D622B");

        return;
    }

    private String getClientIP()
    {
        String forwarded = getHttpRequest().getHeader("X-Forwarded-For");
        if (forwarded == null || forwarded.length() < 3)
        {
            return getHttpRequest().getRemoteAddr();
        }
        //Tools.splitBySemicolon()
        ArrayList<String> IPs = Tools.splitString(forwarded, ",");
        for (int i = 0; i < IPs.size(); i++) {
            String ip = IPs.get(i);
            if (ip != null && !ip.equalsIgnoreCase("unkown") )
            {
                return ip;
            }
        }

        return getHttpRequest().getRemoteAddr();
    }

    public JsonRPCResponseBase exec(JoinGroupRequest request)
    {
        Logger logger = Logger.getLogger(JoinGroup.class);
        System.out.println("JoinGroup begin....");
        JsonRPCResponseBase response = new JsonRPCResponseBase();
        HttpServletRequest r = getHttpRequest();
        Cookie[] cookies = r.getCookies();

        if (cookies == null) {
            response.setStatus(100);
            response.setMessage("no cookie!");
            return response;
        }
        String uin = "";
        String skey = "";
        String access_token = "";
        String openid = "";
        for (int i = 0; i < cookies.length; ++i) {
            String key = cookies[i].getName();
            if (key.equals("access_uin"))
            {
                uin = cookies[i].getValue();
                uin = uin.replaceAll("^o0*", "");
            }
            if (key.equals("access_skey"))
            {
                skey = cookies[i].getValue();
            }
            if (key.equals("access_token"))
            {
                access_token = cookies[i].getValue();
            }
            if (key.equals("access_openid"))
            {
                openid = cookies[i].getValue();
            }
        }
        if (uin.equals("")||skey.equals("")||access_token.equals(""))
        {
            response.setStatus(100);
            response.setMessage("no uin / skey cookie!");
            return response;
        }
        logger.info("access token:"+access_token);
        logger.info("uin:"+uin);
        logger.info("skey:"+skey);
        //检查身份
        try {
            HashMap<String, String> map = Tools.checkAccessToken(access_token);
        }
        catch (IOException e)
        {
            //系统通信本身失败，不能惩罚用户，柔性处理
            logger.error("exception:"+e.getMessage()+e.toString());
            logger.error("system error, jump access token check.");
        }
        catch (NeedLoginException ee)
        {
            //验证access token失败
            ee.printStackTrace();
            response.setStatus(100);
            response.setMessage("access token invalid");

            logger.error("user access token invalid!");
            return response;

        }
        String clientIP = getClientIP();


        String[] groupcodeList = new String[]{
            "384315982",
            "163455904",
            "416460457",
            "417904140",
            "417821180",
            "429312381",
            "229464965",
            "383089217",
            "429549693",
            "419096764",
            "419281306"};
        int index = new Integer(uin).intValue() % 11;

        String grp1 = groupcodeList[index];

        logger.info("the user "+uin+" will be added into "+grp1);


        joinGroup(uin, skey, access_token, clientIP, grp1, openid);


        writeDB(uin, request.getResource(), clientIP);

        response.setMessage("success");
        response.setStatus(0);

        return response;
    }
    /*
毫秒服务引擎(单向push) 243696424
毫秒服务引擎(讨论群) 384315982
毫秒服务引擎备用1 163455904
毫秒服务引擎备用2 416460457
毫秒服务引擎备用3 417904140
毫秒服务引擎备用4 417821180
毫秒服务引擎备用5 429312381
毫秒服务引擎备用6 229464965
毫秒服务引擎备用7 383089217
毫秒服务引擎备用8 429549693
毫秒服务引擎备用9 419096764
毫秒服务引擎备用10 419281306

     */
    static private void joinGroup(String uin, String skey, String access_token, String clientIP, String groupcode, String openid)
    {
        return;
        /*

        try {
            HttpClient client = new HttpClient();
            client.getHttpConnectionManager().getParams().setConnectionTimeout(3000);
            client.getHttpConnectionManager().getParams().setSoTimeout(3000);

            GetMethod method = new GetMethod("http://cface.qq.com/cgi-bin/JoinGroupCGI?access_clientip="+clientIP+"&access_groupcode="+groupcode);
            method.getParams().setCookiePolicy(CookiePolicy.RFC_2109);
            method.setRequestHeader("Cookie",
                    String.format("access_skey=%s;access_uin=%s;access_token=%s;access_appid=101359176;access_openid=%s",
                            skey,
                            uin,
                            access_token,
                            openid));



            int status = client.executeMethod(null, method);
            if (status != HttpStatus.SC_OK) {
                throw new IOException("call graph.qq.com failed");
            }

            byte[] body = method.getResponseBody();
            String jsonStr = new String(body, Charset.forName("utf8"));
            System.out.println(jsonStr);


        }
        catch (IOException  e)
        {
            e.printStackTrace();

        }
        */



    }

    private void writeDB(String uin, String resource, String ip)
    {

        DBUtil util = new DBUtil();
        try {
            if (util.getConnection() == null) {
                throw new Exception("failed to connect to database");
            }
            String sql = "insert into t_DownloadCount(QQ,downloadTime, resource, userIP) values(?,?,?, ?) ";
            List<Object> params = new ArrayList<Object>();

            params.add(uin);
            params.add(Tools.nowString(null));
            params.add(resource);
            params.add(ip);
            util.updateByPreparedStatement(sql, params);
        }
        catch (Exception e)
        {
            e.printStackTrace();

        }
        finally {
            util.releaseConn();
        }
    }

}
