package com.company;

import org.apache.commons.httpclient.URI;
import org.apache.commons.httpclient.methods.PostMethod;
import org.apache.commons.httpclient.methods.PutMethod;
import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.methods.multipart.MultipartRequestEntity;

import java.nio.charset.Charset;
import java.util.Map;

/**
 * Created by Administrator on 2016/12/28.
 */
public class AccessES {

    static public void DoHttpPut(String url,Map<String, String> headers, String requestBody, StringBuffer responseBody) throws Exception
    {
        PutMethod  method = new PutMethod(url);
        HttpClient client = new HttpClient();

        if (headers != null) {
            for (Map.Entry<String, String> entry : headers.entrySet()) {
                method.addRequestHeader(entry.getKey(), entry.getValue());
            }
        }
        method.setRequestBody(requestBody);
        client.executeMethod(method);

        byte[] bodyBytes = method.getResponseBody();
        responseBody.append(new String(bodyBytes, Charset.forName("UTF-8")));

        method.releaseConnection();
    }
    static public void DoHttpPost(String url,Map<String, String> headers, String requestBody, StringBuffer responseBody) throws Exception
    {
        PostMethod method = new PostMethod(url);
        HttpClient client = new HttpClient();

        if (headers != null) {
            for (Map.Entry<String, String> entry : headers.entrySet()) {
                method.addRequestHeader(entry.getKey(), entry.getValue());
            }
        }
        method.setRequestBody(requestBody);
        client.executeMethod(method);

        byte[] bodyBytes = method.getResponseBody();
        responseBody.append(new String(bodyBytes, Charset.forName("UTF-8")));

        method.releaseConnection();
    }

/*
    PutMethod m_put = null;
    HttpClient m_client = new HttpClient();

    static public AccessES getInstance4Put(String host)
    {
        AccessES ret = new AccessES();
        ret.m_client.setConnectionTimeout(300);
        ret.m_client.setTimeout(300);
        ret.m_put = new PutMethod(host);
        return ret;
    }
    public void put(String uriStr,Map<String, String> headers, String requestBody, StringBuffer responseBody) throws Exception
    {

        if (headers != null) {
            for (Map.Entry<String, String> entry : headers.entrySet()) {
                m_put.addRequestHeader(entry.getKey(), entry.getValue());
            }
        }
        m_put.setURI(new URI(uriStr));
        m_put.setRequestBody(requestBody);
        m_client.executeMethod(m_put);

        byte[] bodyBytes = m_put.getResponseBody();
        responseBody.append(new String(bodyBytes, Charset.forName("UTF-8")));


    }
    public  void releaseConnection()
    {
        if (m_put != null) {m_put.releaseConnection();}

    }
    */

}
