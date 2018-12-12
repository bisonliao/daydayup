package ngse.org;

import beans.service.Login;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Created by Administrator on 2016/1/23.
 * 所有ajax后台处理类都继承自该类，该类有几个成员变量，保存了http请求的上下文信息
 * 方便业务流程处理的时候使用
 */
public class JsonRPCHandler {

    private HttpServlet servlet;
    private HttpServletRequest httpRequest;
    private HttpServletResponse httpResponse;

    public String checkIdentity()
    {
        Cookie[] cookies = this.getHttpRequest().getCookies();
        int i;
        String staffName = "";
        String ticket = "";
        for (i = 0; i < cookies.length; ++i)
        {
            Cookie cookie = cookies[i];
            if (cookie.getName().equals("ngse_user"))
            {
                staffName = cookie.getValue();
            }
            if (cookie.getName().equals("ngse_ticket"))
            {
                ticket = cookie.getValue();
            }
        }
        if (staffName.length() < 1 || ticket.length() < 1)
        {
            return "get cookie failed!";
        }
        if (Login.checkTicket(staffName, ticket))
        {
            return "success";
        }
        else
        {
            return "checkTicket() false";
        }

    }

    public HttpServlet getServlet() {
        return servlet;
    }

    public void setServlet(HttpServlet servlet) {
        this.servlet = servlet;
    }

    public HttpServletRequest getHttpRequest() {
        return httpRequest;
    }

    public void setHttpRequest(HttpServletRequest httpRequest) {
        this.httpRequest = httpRequest;
    }

    public HttpServletResponse getHttpResponse() {
        return httpResponse;
    }

    public void setHttpResponse(HttpServletResponse httpResponse) {
        this.httpResponse = httpResponse;
    }
}
