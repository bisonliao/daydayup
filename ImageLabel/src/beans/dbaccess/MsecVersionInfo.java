package beans.dbaccess;

/**
 * Created by Administrator on 2016/10/17.
 */
public class MsecVersionInfo {
    String version;
    String description;
    String binaryUrl;
    String docUrl;

    public MsecVersionInfo()
    {
        version = "";
        description = "";
        binaryUrl = "";
        docUrl = "";
    }
    public MsecVersionInfo(String ver, String desc, String bin, String doc)
    {
        version = ver;
        description = desc;
        binaryUrl = bin;
        docUrl = doc;
    }

    public String getVersion() {
        return version;
    }

    public void setVersion(String version) {
        this.version = version;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getBinaryUrl() {
        return binaryUrl;
    }

    public void setBinaryUrl(String binaryUrl) {
        this.binaryUrl = binaryUrl;
    }

    public String getDocUrl() {
        return docUrl;
    }

    public void setDocUrl(String docUrl) {
        this.docUrl = docUrl;
    }
}
