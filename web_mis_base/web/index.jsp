<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2016/1/25
  Time: 12:48
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head>
    <title></title>
  <link rel="stylesheet" type="text/css" href="/css/MenuTree.css"/>
  <link rel="stylesheet" type="text/css" href="/css/bootstrap.min.css"/>
  <link rel="stylesheet" href="/css/main.css">
  <link rel="stylesheet" href="/js/jquery_ui/jquery-ui.css">
  <link rel="stylesheet" href="/js/jquery_ui/jquery-ui.structure.css">
  <link rel="stylesheet" href="/js/jquery_ui/jquery-ui.theme.css">

  <script type="text/javascript" src="/js/jquery-2.2.0.min.js"></script>
  <script type="text/javascript" src="/js/jquery.form.js"></script>
  <script type="text/javascript" src="/js/jquery.cookie.js"></script>
  <script type="text/javascript" src="/js/jquery_ui/jquery-ui.js"></script>


  <style>
  #main{width:auto;height:auto;     }
  #top_div{width:990px;height:30px;   background-color:#444444  ;float:top ;color:#f0f0f0}
  #left{width:190px;height:1000px;float: left; background-color:#f0f0f0;}
  #right{width:780px;height:1000px ;float:right;margin-left: 10px}
  #left,#right{float:left;}
  </style>

  <script type="text/javascript">
    function showTips(str)
    {
      $("#result_message").attr("title", "message");
      $("#result_message").empty();
      $("#result_message").append(str);

      $("#result_message").dialog();

    };

    $(document).ready(function() {



      $("#left").load("/pages/LeftMenu.jsp");


      var u = $.cookie("ngse_user");
      if (u != null && u.length > 0)
      {

        $("#user_name").empty();

        var str = "&nbsp;&nbsp;&nbsp;&nbsp;<a href='/pages/users/login.jsp'><img src='/imgs/exit.png' alt='logout'/></a>";
        $("#user_name").append(u);
        $("#user_name").append(str);
      }



    })



  </script>
</head>
<body>
<div id="main">
  <div id="top_div">
    <p> &nbsp;&nbsp;redis console&nbsp;&nbsp;&nbsp;&nbsp; <span id="user_name" style="font-style:italic;color:greenyellow;float:right"></span>  </p>
  </div>
  <div id="body_div">
      <div id="left" ></div>
      <div id="right"  ></div>
  </div>

</div>
</body>
</html>
