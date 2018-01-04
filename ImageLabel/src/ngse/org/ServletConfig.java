package ngse.org;



import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.PropertyConfigurator;


import javax.servlet.ServletContext;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.File;
import java.io.IOException;

/**
 * Created by Administrator on 2016/1/26.
 * web console项目不可避免的有些配置信息，需要在tomcat一启动就加载
 * ServletConfig这个 servlet的成员变量保存了这些配置
 * 这些配置写在web.xml文件里
 */
@WebServlet(name = "ServletConfig")
public class ServletConfig extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

    }
    public static String fileServerRootDir = "";//文件传输的根目录
    public static String userTicketKey = "1f1735424446fd5ebf34642c0644fba7";//用户身份票据的加密key，需要保密

    @Override
    public void init(javax.servlet.ServletConfig config) throws ServletException {

        //System.out.println("Log4JInitServlet 正在初始化 log4j日志设置信息");
        String log4jLocation = config.getInitParameter("log4j-properties-location");
        fileServerRootDir = config.getInitParameter("FileServerRootDir");
        if (fileServerRootDir == null)
        {
            fileServerRootDir = "/home/files";
        }
        userTicketKey = config.getInitParameter("UserTicketKey");
        if (userTicketKey == null)
        {
            fileServerRootDir = "1f1735424446fd5ebf34642c0644fba7";
        }


        ServletContext sc = config.getServletContext();

        JsTea.context = sc;

        if (log4jLocation == null) {
            System.err.println("*** 没有 log4j-properties-location 初始化的文件, 所以使用 BasicConfigurator初始化");
            BasicConfigurator.configure();
        } else {
            String webAppPath = sc.getRealPath("/");
            String log4jProp = webAppPath + log4jLocation;
            File yoMamaYesThisSaysYoMama = new File(log4jProp);
            if (yoMamaYesThisSaysYoMama.exists()) {
                //System.out.println("使用: " + log4jProp+"初始化日志设置信息");
                PropertyConfigurator.configure(log4jProp);
            } else {
                System.err.println("*** " + log4jProp + " 文件没有找到， 所以使用 BasicConfigurator初始化");
                BasicConfigurator.configure();
            }
        }





        super.init(config);
    }


}
