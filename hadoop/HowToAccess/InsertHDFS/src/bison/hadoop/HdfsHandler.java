package bison.hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;

import java.io.IOException;
import java.net.URI;

/**
 * Created by Administrator on 2016/11/23.
 */
public class HdfsHandler {
    private Configuration configuration;
    private FileSystem fileSystem;
    private FSDataInputStream inputStream;
    private FSDataOutputStream outputStream;

    private HdfsHandler()
    {
        return;
    }
    static public HdfsHandler openFileForRead(String filename) throws Exception
    {
        HdfsHandler handler = new HdfsHandler();
        handler.configuration = new Configuration();
        handler.fileSystem = FileSystem.get(URI.create(filename), handler.configuration);
        handler.inputStream = handler.fileSystem.open(new Path(filename));
        handler.outputStream = null;
        return handler;
    }
    static public HdfsHandler openFileForWrite(String filename, short repNum) throws Exception
    {
        HdfsHandler handler = new HdfsHandler();
        handler.configuration = new Configuration();

        Path path = new Path(filename);
        handler.fileSystem = FileSystem.get(URI.create(filename), handler.configuration);

        if (handler.fileSystem.exists(path))
        {
            handler.outputStream = handler.fileSystem.append(path);
        }
        else {
            long blockSize = 64*1024*1024;
            handler.outputStream = handler.fileSystem.create(new Path(filename),
                    false,
                    100*1024,
                    repNum,
                    blockSize);
        }
        handler.inputStream = null;
        return handler;
    }

    static public boolean fileExist(String path) throws Exception
    {
        FileSystem fileSystem = null;
        boolean result = false;
        try {


            Configuration configuration = new Configuration();
            fileSystem = FileSystem.get(URI.create(path), configuration);

            result = fileSystem.exists(new Path(path));
        }
        finally {
            if (fileSystem != null) {fileSystem.close();}
        }

        return result;

    }
    static public boolean isDirectory(String path) throws Exception
    {
        FileSystem fileSystem = null;
        boolean result = false;
        try {


            Configuration configuration = new Configuration();
            fileSystem = FileSystem.get(URI.create(path), configuration);

            result = fileSystem.isDirectory(new Path(path));
        }
        finally {
            if (fileSystem != null) {fileSystem.close();}
        }

        return result;
    }
    static public void mkdir(String path) throws Exception
    {
        FileSystem fileSystem = null;
        try {


            Configuration configuration = new Configuration();
            fileSystem = FileSystem.get(URI.create(path), configuration);

            fileSystem.mkdirs(new Path(path));
        }
        finally {
            if (fileSystem != null) {fileSystem.close();}
        }

    }
    static public void delete(String path) throws Exception
    {
        FileSystem fileSystem = null;
        try {


            Configuration configuration = new Configuration();
            fileSystem = FileSystem.get(URI.create(path), configuration);

            fileSystem.delete(new Path(path), true);
        }
        finally {
            if (fileSystem != null) {fileSystem.close();}
        }

    }
    public int readBytes(byte[] buf) throws Exception
    {
        if (inputStream == null)
        {
            throw new IOException("handler has not been opened!");
        }
        return inputStream.read(buf);
    }
    public void writeBytes(byte[] buf, int offset, int len) throws Exception
    {
        if (outputStream == null)
        {
            throw new IOException("handler has not been opened!");
        }
        outputStream.write(buf, offset, len);
    }
    public void close() throws Exception
    {

        if (inputStream != null) {
            inputStream.close();
        }
        if (outputStream != null) {
            outputStream.close();
        }
        if (fileSystem != null) {
            fileSystem.close();
        }

    }

}
