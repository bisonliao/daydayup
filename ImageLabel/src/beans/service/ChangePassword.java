package beans.service;

import beans.dbaccess.StaffInfo;
import beans.request.LoginRequest;

import beans.response.LoginResponse;
import ngse.org.*;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * Created by Administrator on 2016/2/10.
 * 用户登录
 */
public class ChangePassword extends JsonRPCHandler {

    private String geneTicket(String userName)
    {

        String userNameMd5 =  Tools.md5(userName);//32bytes
        String dt = String.format("%16d", new Date().getTime()/1000);


        String plainText = userNameMd5+dt; // 48bytes

        JsTea tea = new JsTea(null);
        return tea.encrypt(plainText, ServletConfig.userTicketKey);//96bytes
    }
    static public  boolean  checkTicket(String userName, String ticket)
    {
        if (ticket.length() != 96) { return false;}

        JsTea tea = new JsTea(null);
        String s = tea.decrypt(ticket, ServletConfig.userTicketKey);
        if (s.length() != 48) { return false;}
        String userNameMd5 = s.substring(0, 32);
        String dt = s.substring(32, 48);
        long ticketInitTime = new Integer(dt.trim()).intValue();
        long currentTime = new Date().getTime()/1000;
        if (ticketInitTime < currentTime && (currentTime-ticketInitTime) > (3600*24))
        {
            return false;
        }
        String md5Str = Tools.md5(userName);
        if (md5Str.equals(userNameMd5))
        {
            return true;
        }
        else
        {
            return false;
        }

    }


     public JsonRPCResponseBase exec(LoginRequest request)
     {
         JsonRPCResponseBase resp = new JsonRPCResponseBase();

         if (request.getStaff_name() == null && request.getTgt() == null||
                 request.getNew_password() == null)
         {
             resp.setStatus(100);
             resp.setMessage("login name /password empty!");
             return resp;
         }



         DBUtil util = new DBUtil();
         if (util.getConnection() == null)
         {
             resp.setStatus(100);
             resp.setMessage("db connect failed!");
             return resp;
         }
         List<StaffInfo> staffInfoList ;




         try {
             String sql = "select staff_name, staff_phone,password,salt from t_staff "+
                     " where  staff_name=? ";
             List<Object> params = new ArrayList<Object>();
             params.add(request.getStaff_name());

             staffInfoList = util.findMoreRefResult(sql, params, StaffInfo.class);
             if (staffInfoList.size() != 1)
             {
                 resp.setMessage("user does NOT exist.");
                 resp.setStatus(100);
                 return resp;
             }
             //用加盐的二次密码hash作为key（数据库里存着）解密
             StaffInfo staffInfo = staffInfoList.get(0);
             JsTea tea = new JsTea(this.getServlet());
             String p1 = tea.decrypt(request.getTgt(), staffInfo.getPassword());
             ///获取session里保存的challenge
             String challenge = (String)(getHttpRequest().getSession().getAttribute(GetSalt.CHALLENGE_KEY_IN_SESSION));
             if (p1.length() != 40 )
             {
                 resp.setMessage("p1 error!");
                 resp.setStatus(100);
                 return resp;
             }
             //看解密处理的后面部分内容是否同challenge，放重放
             if (!p1.substring(32).equals(challenge))
             {
                 resp.setMessage("p1 error!!");
                 resp.setStatus(100);
                 return resp;
             }
            //根据解密出来的一次密码hash，现场生成二次加盐的hash，与数据库里保存的比较看是否相等
             String p2 = AddNewStaff.geneSaltedPwd(p1.substring(0, 32), staffInfo.getSalt());
             if (!p2.equals(staffInfo.getPassword()))
             {
                 resp.setMessage("p1 error!!!");
                 resp.setStatus(100);
                 return resp;
             }
             //当前密码验证成功，开始改密
             sql = "update t_staff set password=? where staff_name=?";
             params = new ArrayList<Object>();
             params.add(request.getNew_password());
             params.add(request.getStaff_name());

             int updateNum = util.updateByPreparedStatement(sql, params);
             if (updateNum != 1)
             {
                 resp.setMessage("update password failed");
                 resp.setStatus(100);
                 return resp;
             }

             resp.setMessage("success");
             resp.setStatus(0);
             return resp;




         }
         catch (Exception e)
         {
             resp.setStatus(100);
             resp.setMessage("db query exception!");
             e.printStackTrace();
             return resp;
         }
         finally {
             util.releaseConn();
         }



     }
}
