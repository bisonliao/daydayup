<%--
  Created by IntelliJ IDEA.
  User: Administrator
  Date: 2016/1/25
  Time: 14:07
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head>
    <title></title>

  <style>
    #title_bar{width:auto;height:auto;   margin-top: 5px }
    #query_form{width:auto;height:auto;  padding:20px;margin-top: 5px;}
    #result_message{width:auto;height:auto;  margin-top: 5px;}

  </style>
  <script type="text/javascript">
      function LevelChanged()
      {
          var level = $("#service_level").val();
          $("#div_parent").toggle();
          if (level == "first_level")
          {
              $("#service_parent").attr("disabled", true);



          }
          else
          {

             $("#service_parent").attr("disabled", false);



          }
      }

      function LoadAvaibleFirstsLevelService()
      {
          var   request={
              "handleClass":"beans.service.QueryFirstLevelService",
              "requestBody": {"service_name": null},
          };
          $.post("/JsonRPCServlet",
                  {request_string:JSON.stringify(request)},

                  function(data, status){
                      if (status == "success") {//http通信返回200
                          if (data.status == 0) {//业务处理成功

                              console.log(data.service_list);
                              var str = "";
                              $.each(data.service_list, function (i, rec) {
                                  str += "<option value =\""+rec.first_level_service_name+"\">"+rec.first_level_service_name+"</option>\r\n";
                                  //<option value ="1">Volvo</option>

                              });
                              console.log(str);
                              $("#service_parent").empty();
                              $(str).appendTo("#service_parent");


                          }
                          else if (data.status == 99)
                          {
                              document.location.href = "/pages/users/login.jsp";
                          }
                          else
                          {
                              //showTips(data.message);//业务处理失败的信息
                              var str="<p>"+"加载一级服务名异常："+data.message+"</p>";

                              $("#result_message").empty();
                              $(str).appendTo("#result_message");
                          }
                      }
                      else
                      {
                          //showTips(status);//http通信失败的信息
                          var str="<p>"+"加载一级服务名异常："+status+"</p>";

                          $("#result_message").empty();
                          $(str).appendTo("#result_message");
                      }

                  });

      }


      function onAddBtnClicked()
      {
          var   service_name= $("#service_name").val();
          var   service_level= $("#service_level").val();
          var   service_parent= $("#service_parent").val();
          if (service_name == null || service_name.length < 1)
          {
              showTips("服务名不能为空");
              return;
          }
          if (!/^[a-zA-Z][a-zA-Z_0-9]+$/.test(service_name))
          {
              showTips("服务名只能由字母数字和下划线组成，且必须字母开头");
              return;
          }
          if (service_name == "agent" && service_level == "first_level")
          {
              showTips("服务名'agent'不合适，请改用其他名字。");
              return;
          }

          var   request={
              "handleClass":"beans.service.AddService",
              "requestBody": {"service_name": service_name,
                  "service_level":service_level,
                  "service_parent": service_parent},
          };
          $.post("/JsonRPCServlet",
                  {request_string:JSON.stringify(request)},

                  function(data, status){
                      if (status == "success") {//http通信返回200
                          if (data.status == 0) {//业务处理成功
                              $("#left").load("/pages/LeftMenu.jsp");
                              showTips("添加成功。请从左侧树状菜单进入IP等进一步的配置");
                              LoadAvaibleFirstsLevelService();

                          }
                          else if (data.status == 99)
                          {
                              document.location.href = "/pages/users/login.jsp";
                          }
                          else
                          {
                              showTips(data.message);
                              //showTips(data.message);//业务处理失败的信息

                          }
                      }
                      else
                      {
                          showTips(status);//http通信失败的信息
                          showTips(status);
                      }

                  });

      }
      function onDelBtnClicked()
      {

          var   service_name= $("#service_name").val();
          var   service_level= $("#service_level").val();
          var   service_parent= $("#service_parent").val();
          if (service_name == null || service_name.length < 1)
          {
              showTips("服务名不能为空");
              return;
          }
          console.log("delete "+service_name);
          if (confirm("Delete"+service_name+"?") == false)
          {
              return;
          }

          var   request={
              "handleClass":"beans.service.DelService",
              "requestBody": {"service_name": service_name,
                  "service_level":service_level,
                  "service_parent": service_parent},
          };
          $.post("/JsonRPCServlet",
                  {request_string:JSON.stringify(request)},

                  function(data, status){
                      if (status == "success") {//http通信返回200
                          if (data.status == 0) {//业务处理成功
                              $("#left").load("/pages/LeftMenu.jsp");
                              showTips("删除成功。");
                              LoadAvaibleFirstsLevelService();

                          }
                          else if (data.status == 99)
                          {
                              document.location.href = "/pages/users/login.jsp";
                          }
                          else
                          {
                              showTips(data.message);
                              //showTips(data.message);//业务处理失败的信息

                          }
                      }
                      else
                      {
                          showTips(status);//http通信失败的信息
                          showTips(status);
                      }

                  });

      }

    $(document).ready(function(){


        LoadAvaibleFirstsLevelService();
        $("#div_parent").toggle();




    })




  </script>
</head>
<body>

<div id="title_bar" class="title_bar_style">
  <p>&nbsp;&nbsp;增/删标准服务</p>
</div>

<div id="query_form" class="form_style">
<form class="form-inline">
  <div class="form-group">

    <label for="service_name">服务名:</label>
    <input type="text" class="form-control" id="service_name" maxlength="16">
  </div>
  <div class="form-group">
    <label for="service_level">级别:</label>
      <select class="form-control" id="service_level" onchange="LevelChanged()">
          <option value ="first_level">一级</option>    <option value ="second_level">二级</option>
      </select>
  </div>
    <br>
    <div class="form-group" id="div_parent">
        <label for="service_parent"  >从属上一级服务:</label>
        <select class="form-control" id="service_parent"> </select>
    </div>
    <br>
  <button type="button" class="btn-small" id="btn_add" onclick="onAddBtnClicked()">增加</button>
    <button type="button" class="btn-small" id="btn_del" onclick="onDelBtnClicked()">删除</button>


</form>

</div>

<div id="result_message"  class="result_msg_style">
  </div>
</body>
</html>
