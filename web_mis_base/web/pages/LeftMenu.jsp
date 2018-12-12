<%@ page import="ngse.org.DBUtil" %>
<%@ page import="java.util.List" %>
<%@ page import="java.util.ArrayList" %>
<%@ page import="beans.dbaccess.FirstLevelService" %>
<%@ page import="beans.dbaccess.SecondLevelService" %>
<%@ page import="beans.service.Login" %>
<%@ page import="java.io.PrintWriter" %>
<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2016/1/25
  Time: 10:12
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>

<html>
<head>
  <title>
  </title>
  <link rel="stylesheet" type="text/css" href="/css/MenuTree.css"/>

  <script type="text/javascript" src="/js/MenuTree.js"></script>
  <script type="text/javascript">
    $(function(){
      $(".menu_tree").MenuTree({
        click:function(a){
          //  showTips($(a).attr("ref"));
          if ( $(a).attr("ref") == "leaf" || $(a).attr("ref") =="action" )
          {
            var dest = $(a).attr("href");
              $("#right").html("");
              $("#right").load(dest);


          }
        }
      });




    });
  </script>

    <%
        if (!checkIdentity(request).equals("success"))
        {
            PrintWriter outf = response.getWriter();

            outf.printf("<script type=\"text/javascript\">");
            outf.printf("document.location.href = \"/pages/users/login.jsp\";");
            outf.printf("</script>");


        }
    %>
</head>
<body>
<%! public   String printService()
  {
      StringBuffer outstr = new StringBuffer();
      //我知道把后台java代码写在jsp页面里好恶心噻，但这里页面呈现和逻辑不是很好分离(其实是我js不熟悉不知道怎么处理复杂的json数据)，先忍受一下
      DBUtil jdbcUtils = new DBUtil();
      jdbcUtils.getConnection();
      String sql = "select first_level_service_name from t_first_level_service  where type='standard'";
      List<Object> params = null;
      params = new ArrayList<Object>();
      //params.add("Wtlogin");
      FirstLevelService service;
      SecondLevelService sec_service;
      try {
          List<FirstLevelService> list = jdbcUtils.findMoreRefResult(sql, params, FirstLevelService.class);
          for (int i = 0; i < list.size(); i++) {
              service = list.get(i);
              outstr.append("<li><a href=\"#\" ref=\"\">" + service.getFirst_level_service_name() + "</a></li>\r\n");

              params = new ArrayList<Object>();
              params.add(service.getFirst_level_service_name());
              sql = "select first_level_service_name, second_level_service_name " +
                      "from t_second_level_service where first_level_service_name=? and type='standard'";
              List<SecondLevelService> sub_list = jdbcUtils.findMoreRefResult(sql, params, SecondLevelService.class);
              if (sub_list.size() > 0) {
                  outstr.append("<ul>\r\n");
                  for (int j = 0; j < sub_list.size(); j++) {
                      sec_service = sub_list.get(j);
                      String svc_name = sec_service.getSecond_level_service_name();
                      outstr.append("<li><a href=\"/pages/ShowService.jsp?service=" + svc_name + "&service_parent="+service.getFirst_level_service_name()+"\" ref=\"leaf\">" + svc_name + "</a></li>\r\n");

                  }
                  outstr.append("</ul>\r\n");

              }

          }
          return outstr.toString();
      } catch (Exception e) {

          e.printStackTrace();
          return "";
      }

  }
%>

<%! public   String checkIdentity(HttpServletRequest req)
{
    Cookie[] cookies = req.getCookies();
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

%>




<div class="menu_tree">
    <ul>
      <li><a href="#" ref="">redis实例</a></li>
        <ul>
           <li><a href="/pages/StandardServiceMaintenance.jsp" ref="leaf">【增/删实例】</a></li>
            <%=printService()%>
        </ul>


<!--这部分是静态的-->

        <li><a href="#" ref="">信息管理</a></li>
        <ul>
            <li><a href="/pages/ShowDevOp.jsp" ref="leaf">用户</a></li>
            <li><a href="/pages/users/ChangePasswd.jsp" ref="leaf">改密</a></li>
        </ul>
        <li><a href="#" ref="">帮助</a></li>
        <ul>
            <li><a href="/pages/about.jsp" ref="leaf">关于</a></li>
            <li><a href="/pages/document.jsp" ref="leaf">文档</a></li>
        </ul>

</ul>
</div>

</body>
</html>
