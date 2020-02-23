jarfiles=`ls libs/*.jar`
ff=""
for file in $jarfiles
do
        if [ "$ff" = "" ]; then
        ff=$file
        else
        ff=$ff":"$file
        fi
done

java -cp $ff:dubboDemo-1.0-SNAPSHOT-provider.jar org.apache.dubbo.demo.provider.Provider
