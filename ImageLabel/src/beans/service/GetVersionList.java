package beans.service;

import beans.dbaccess.MsecVersionInfo;
import beans.request.GetVersionListRequest;
import beans.response.GetVersionListResponse;


import ngse.org.DBUtil;
import ngse.org.JsonRPCHandler;
import ngse.org.NeedLoginException;
import ngse.org.Tools;
import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.HttpState;
import org.apache.commons.httpclient.HttpStatus;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.log4j.Logger;
import org.json.JSONObject;


import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by Administrator on 2016/10/17.
 */
public class GetVersionList extends JsonRPCHandler {

    /*
    static public void main(String args[]) throws Exception
    {


       String access_token = "abcdefghijklmn";
        HttpClient client = new HttpClient();

        GetMethod method = new GetMethod("https://graph.qq.com/oauth2.0/me?access_token="+access_token);
        int status = client.executeMethod(null, method);
        if (status != HttpStatus.SC_OK)
        {
            throw  new Exception("call graph.qq.com failed");
        }

        byte[] body = method.getResponseBody();
        String jsonStr = new String(body, Charset.forName("utf8"));
        System.out.println("jsonStr:"+jsonStr);
        Pattern p = Pattern.compile("\\{.*\\}");
        Matcher m = p.matcher(jsonStr);
        if (!m.find())
        {
            throw new Exception("graph.qq.com returns invalid json string.");
        }
        jsonStr = m.group(0);

        JSONObject object = new JSONObject(jsonStr);
        String client_id = object.getString("open_id");
        String open_id = object.getString("openid");

        HashMap<String, String> map = new HashMap<>();
        map.put("client_id", client_id);
        map.put("openid", open_id);
        map.put("access_token", access_token);
    }
    */


    public GetVersionListResponse exec(GetVersionListRequest request)
    {
        GetVersionListResponse response = new GetVersionListResponse();
        Logger logger = Logger.getLogger(GetVersionList.class);

        try {
            HashMap<String, String> map = checkUserLogin(request);
            logger.info("checkUserLogin ok, uin="+map.get("access_uin"));
            String openid = map.get("openid");
            logger.info("check access token ok, openid="+openid);
            Cookie cookie = new Cookie("access_openid",  openid);
           // cookie.setMaxAge(3600*2);
            getHttpResponse().addCookie(cookie);

            writeUserInfoIntoDB(map);
            logger.info("writeUserInfoIntoDB ok,uin="+map.get("access_uin"));
        }
        catch (NeedLoginException e)
        {
            logger.error(e.getMessage());
            response.setMessage(e.getMessage());
            response.setStatus(99);
            return response;
        }


        List<MsecVersionInfo> verList = getVersionListFromDB();
        logger.info("getVersionListFromDB ok, size="+verList.size());
        response.setVersionInfoList(verList);
        response.setStatus(0);
        response.setMessage("success");



        return response;
    }

    private List<MsecVersionInfo> getVersionListFromDB()
    {

        DBUtil util = new DBUtil();
        try {
            if (util.getConnection() == null) {
              throw new Exception("failed to connect to database");
            }
            String sql = "select version, description, binaryUrl, docUrl from t_MsecVersionInfo";
            return util.findMoreRefResult(sql, null, MsecVersionInfo.class);
        }
        catch (Exception e)
        {
            e.printStackTrace();


            //至少给个默认下载的东东
            List<MsecVersionInfo> infoList = new ArrayList<>();

            infoList.add(new MsecVersionInfo("version 1.0（2016.9.30）",
                    "",
                    "/download/msec_1.0.tar",
                    "/download/msec_doc_1.0.zip"));
            return infoList;

        }
        finally {
            util.releaseConn();
        }

    }
    private void writeUserInfoIntoDB(HashMap<String, String> map)
    {

        DBUtil util = new DBUtil();
        try {
            if (util.getConnection() == null) {
                throw new Exception("failed to connect to database");
            }
            String sql = "insert into t_UserInfo(openid,QQ,downloadTime) values(?,?,?) ";
            List<Object> params = new ArrayList<Object>();
            params.add(map.get("access_token"));
            params.add(map.get("access_uin"));
            params.add(Tools.nowString(null));
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

    private HashMap<String, String>  checkUserLogin(GetVersionListRequest jsonRequest) throws NeedLoginException
    {
        String access_token = "";
        String uin = "";

        Logger logger = Logger.getLogger(GetVersionList.class);

        HttpServletRequest r = getHttpRequest();
        Cookie[] cookies = r.getCookies();

        if (cookies == null) {
            throw new NeedLoginException("no auth info in cookie");
        }
        for (int i = 0; i < cookies.length; ++i) {
            String key = cookies[i].getName();
            //logger.info("cookie:"+key+":"+cookies[i].getValue());
            if (key.equals("access_token"))
            {
                access_token = cookies[i].getValue();
            }
            if (key.equals("access_uin"))
            {
                uin = cookies[i].getValue();
            }
        }

        if (access_token == null || access_token.length() < 10)
        {
            logger.info("no auth info");
            throw new NeedLoginException("no auth info");
        }
        HashMap<String, String> map = null;
        try {
            map = Tools.checkAccessToken(access_token);
        }
        catch (IOException e)
        {
            logger.error("exception:"+e.getMessage()+e.toString());
            logger.error("system error, jump access token check.");
            //通信本身失败，不能惩罚用户
            map = new HashMap<>();
            map.put("client_id", "");
            map.put("openid", "");
            map.put("access_token", access_token);
            map.put("access_uin", uin);


            return map ;

        }


        map.put("access_token", access_token);
        map.put("access_uin", uin);


        return map ;
    }
}
