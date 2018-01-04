package image.label;

import javafx.util.Pair;

import javax.servlet.http.HttpSession;
import java.io.*;
import java.nio.file.FileSystem;
import java.util.*;

/**
 * Created by bisonliao on 2017/12/29.
 */
public class Worker {
    private String workerID;
    private Map<String, Integer> imageLabeled; //已经明确标注
    private Set<String> imageDeleted; //已经明确删除
    private Map<String, Integer> imageToBeLabeled; //待标注
    private Iterator<Map.Entry<String, Integer>> iterateToBeLabeled;

    private void startEpoch()
    {
        //剔除掉已经明确的image
        for (String key : imageLabeled.keySet())
        {
            imageToBeLabeled.remove(key);
        }
        for (String key : imageDeleted)
        {
            imageToBeLabeled.remove(key);
        }
        iterateToBeLabeled = imageToBeLabeled.entrySet().iterator();
    }


    public Worker(String id)
    {
        workerID = id;
        imageLabeled = new HashMap<String, Integer>();
        imageDeleted = new HashSet<String>();
        imageToBeLabeled = new HashMap<>();
        String filename;
        File file ;

        // labeled image
        filename = Environment.WEB_ROOT + File.separator + "data"+File.separator+"worker" +
                File.separator + workerID + "_confirmed.txt";
        file = new File(filename);
        if (file.exists())
        {
            try {
                BufferedReader in = new BufferedReader(new FileReader(file));
                String line;
                while ( (line = in.readLine()) != null )
                {
                    int pos = line.indexOf(" ");
                    if (pos < -1 || pos >= line.length())
                    {
                        continue;
                    }
                    String imageFilename = line.substring(0, pos);
                    Integer label = new Integer(line.substring(pos+1));

                    imageLabeled.put(imageFilename, label);
                }
                in.close();

            }
            catch (Exception e)
            {
                e.printStackTrace();
            }

        }
        System.out.println("confirmed samples:"+imageLabeled.size());

        // deleted image
        filename = Environment.WEB_ROOT + File.separator + "data"+File.separator+"worker" +
                File.separator + workerID + "_deleted.txt";
        file = new File(filename);
        if (file.exists())
        {
            try {
                BufferedReader in = new BufferedReader(new FileReader(file));
                String line;
                while ( (line = in.readLine()) != null )
                {
                    deleteImage(line);
                }
                in.close();

            }
            catch (Exception e)
            {
                e.printStackTrace();
            }
        }
        System.out.println("deleted samples:"+imageDeleted.size());


        //  images to be labeled
        filename = Environment.WEB_ROOT + File.separator + "data" + File.separator +"worker" +
            File.separator + workerID + ".txt";
        file = new File(filename);
        if (file.exists())
        {
            try {
                BufferedReader in = new BufferedReader(new FileReader(file));
                String line;
                while ( (line = in.readLine()) != null )
                {
                    int pos = line.indexOf(" ");
                    if (pos < -1 || pos >= line.length())
                    {
                        continue;
                    }
                    String imageFilename = line.substring(0, pos);
                    Integer label = new Integer(line.substring(pos+1));

                    imageToBeLabeled.put(imageFilename, label);
                }
                in.close();

            }
            catch (Exception e)
            {
                e.printStackTrace();
            }

        }

        startEpoch();
        System.out.println("to comfirm :"+imageToBeLabeled.size());



    }

    public void confirmLabel(String imageFilename, int label)
    {
        imageLabeled.put(imageFilename, label);
    }

    public void deleteImage(String imageFilename)
    {
        imageDeleted.add(imageFilename);
    }
    public void dump() throws Exception
    {
        String filename;
        File file;

        filename = Environment.WEB_ROOT + File.separator + "data"+File.separator+"worker" + File.separator + workerID+"_confirmed.txt";
        file = new File(filename);

        FileWriter out = new FileWriter(file);
        for (String key : imageLabeled.keySet())
        {
            out.write(String.format("%s %d\n", key, imageLabeled.get(key).intValue()));
        }
        out.close();

        filename = Environment.WEB_ROOT + File.separator + "data"+File.separator+"worker"  + File.separator + workerID+"_deleted.txt";
        file = new File(filename);

        out = new FileWriter(file);
        for (String key : imageDeleted)
        {
            out.write(String.format("%s\n", key));
        }
        out.close();
    }
    public ImageLabelPair getNext()
    {
        if (!iterateToBeLabeled.hasNext())//搞完一轮已经到底
        {
            startEpoch();
        }
        if (iterateToBeLabeled.hasNext())
        {
            Map.Entry<String, Integer> entry = iterateToBeLabeled.next();
            ImageLabelPair pair = new ImageLabelPair();
            pair.setImageFilename(entry.getKey());
            pair.setLabel( entry.getValue().intValue());
            return pair;

        }
        else
        {
            return null;
        }
    }
    public List<String> getLabelTextList()
    {
        String filename = Environment.WEB_ROOT + File.separator + "data" + File.separator + "labels.txt";
        File file = new File(filename);
        List<String> list = new ArrayList<>();
        if (file.exists())
        {
            try {
                BufferedReader in = new BufferedReader(new FileReader(file));
                String line;
                while ( (line = in.readLine()) != null )
                {
                    /*
                    int pos = line.indexOf(" ");
                    if (pos < -1 || pos >= line.length())
                    {
                        continue;
                    }

                    String labelText = line.substring(pos+1);
                    */
                    if (line.length() > 1 && line.charAt(0) == '#')
                    {
                        continue;
                    }
                    list.add(line);
                }
                in.close();

            }
            catch (Exception e)
            {
                e.printStackTrace();
            }

        }
        return list;

    }
}
