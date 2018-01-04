package beans.dbaccess;

/**
 * Created by Administrator on 2016/1/27.
 */
public class DBAnalyzeInfo {
    Long record_number;
    Integer max_integer;
    Integer min_integer;

    public Long getRecord_number() {
        return record_number;
    }

    public void setRecord_number(Long record_number) {
        this.record_number = record_number;
    }

    public Integer getMax_integer() {
        return max_integer;
    }

    public void setMax_integer(Integer max_integer) {
        this.max_integer = max_integer;
    }

    public Integer getMin_integer() {
        return min_integer;
    }

    public void setMin_integer(Integer min_integer) {
        this.min_integer = min_integer;
    }
}
