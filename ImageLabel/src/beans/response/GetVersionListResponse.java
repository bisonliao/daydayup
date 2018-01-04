package beans.response;

import beans.dbaccess.MsecVersionInfo;
import ngse.org.JsonRPCResponseBase;

import java.util.List;

/**
 * Created by Administrator on 2016/10/17.
 */
public class GetVersionListResponse extends JsonRPCResponseBase {
    List<MsecVersionInfo> versionInfoList;

    public List<MsecVersionInfo> getVersionInfoList() {
        return versionInfoList;
    }

    public void setVersionInfoList(List<MsecVersionInfo> versionInfoList) {
        this.versionInfoList = versionInfoList;
    }
}
