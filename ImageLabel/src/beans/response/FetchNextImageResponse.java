package beans.response;

import ngse.org.JsonRPCResponseBase;

import java.util.List;

/**
 * Created by bisonliao on 2017/12/29.
 */
public class FetchNextImageResponse extends JsonRPCResponseBase {
    String imageFilename;
    List<String> labelTextList;
    Integer label;

    public String getImageFilename() {
        return imageFilename;
    }

    public void setImageFilename(String imageFilename) {
        this.imageFilename = imageFilename;
    }

    public List<String> getLabelTextList() {
        return labelTextList;
    }

    public void setLabelTextList(List<String> labelTextList) {
        this.labelTextList = labelTextList;
    }

    public Integer getLabel() {
        return label;
    }

    public void setLabel(Integer label) {
        this.label = label;
    }
}
