package ngse.org;

import org.apache.commons.fileupload.FileItem;
import org.apache.commons.fileupload.FileUploadException;
import org.apache.commons.fileupload.disk.DiskFileItemFactory;
import org.apache.commons.fileupload.servlet.ServletFileUpload;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

/**
 * Created by Administrator on 2016/2/22.
 */
@WebServlet(name = "FileUploadServlet")
public class FileUploadServlet extends HttpServlet {

    //处理multi-part格式的http请求
    //将key-value字段放到fields里返回
    //将文件保存到tmp目录，并将文件名保存到filesOnServer列表里返回
    static protected String FileUpload( Map<String, String> fields, List<String> filesOnServer,
                                     HttpServletRequest request, HttpServletResponse response)
    {

        boolean isMultipart = ServletFileUpload.isMultipartContent(request);
        // Create a factory for disk-based file items
        DiskFileItemFactory factory = new DiskFileItemFactory();
        int MaxMemorySize = 10000000;
        int MaxRequestSize = MaxMemorySize;
        String tmpDir = System.getProperty("TMP", "/tmp");
        //System.out.printf("temporary directory:%s", tmpDir);

// Set factory constraints
        factory.setSizeThreshold(MaxMemorySize);
        factory.setRepository(new File(tmpDir));

// Create a new file upload handler
        ServletFileUpload upload = new ServletFileUpload(factory);
        upload.setHeaderEncoding("utf8");


// Set overall request size constraint
        upload.setSizeMax(MaxRequestSize);

// Parse the request
        try {
            List<FileItem> items = upload.parseRequest(request);
            // Process the uploaded items
            Iterator<FileItem> iter = items.iterator();
            while (iter.hasNext()) {
                FileItem item = iter.next();
                if (item.isFormField()) {//普通的k -v字段

                    String name = item.getFieldName();
                    String value = item.getString("utf-8");
                    fields.put(name, value);
                }
                else {

                    String fieldName = item.getFieldName();
                    String fileName = item.getName();
                    if (fileName == null || fileName.length() < 1)
                    {
                        return "file name is empty.";
                    }
                    String localFileName = ServletConfig.fileServerRootDir+File.separator+"tmp"+File.separator+fileName;
                    //System.out.printf("upload file:%s", localFileName);
                    String contentType = item.getContentType();
                    boolean isInMemory = item.isInMemory();
                    long sizeInBytes = item.getSize();
                    File uploadedFile = new File(localFileName);
                    item.write(uploadedFile);
                    filesOnServer.add(localFileName);
                }


            }
            return "success";
        }catch (FileUploadException e)
        {
            e.printStackTrace();
            return e.toString();
        }
        catch (Exception e)
        {
            e.printStackTrace();
            return e.toString();
        }

    }
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {


        Map<String, String> fields = new HashMap<String, String>();
        List<String> fileNames = new ArrayList<String>();

        request.setCharacterEncoding("UTF-8");

        String result = FileUpload(fields, fileNames, request, response);

        response.setCharacterEncoding("UTF-8");
        response.setContentType("text/html; charset=utf-8");
        PrintWriter out =  response.getWriter();

        if (result == null || !result.equals("success"))
        {
            out.printf("{\"status\":100, \"message\":\"%s\"}", result == null? "":result);
            return;
        }
        String handleClass = fields.get("handleClass");
        if (handleClass != null && handleClass.equals("beans.service.LibraryFileUpload"))
        {
            //out.write(new LibraryFileUpload().run(fields, fileNames));
            return;
        }
        if (handleClass != null && handleClass.equals("beans.service.SharedobjectUpload"))
        {
          //  out.write(new SharedobjectUpload().run(fields, fileNames));
            return;

        }
        out.write("{\"status\":100, \"message\":\"unkown handle class\"}");
        return;


    }


}
