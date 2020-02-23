package org.apache.dubbo.demo.consumer;

import org.springframework.context.support.ClassPathXmlApplicationContext;
import org.apache.dubbo.demo.intf.DemoService;
import org.apache.dubbo.config.ApplicationConfig;
import org.apache.dubbo.config.ReferenceConfig;
import org.apache.dubbo.config.RegistryConfig;
import org.apache.dubbo.config.ProviderConfig;
import org.apache.dubbo.config.ServiceConfig;
import org.apache.dubbo.config.ProtocolConfig;


public class Consumer {
    public static void main(String[] args) throws Exception {
        ClassPathXmlApplicationContext context = new ClassPathXmlApplicationContext(new String[] {"consumer.xml"});
        context.start();
        DemoService demoService = (DemoService)context.getBean("demoService"); // 获取远程服务代理
        for (int i = 0; i < 10; i++) {
            String hello = demoService.sayHello("world#"+i); // 执行远程方法
            System.out.println(hello); // 显示调用结果
            try
            {
                Thread.sleep(1000, 0);
            }
            catch (Exception ee) {}
        }

/*
// 当前应用配置
ApplicationConfig application = new ApplicationConfig();
application.setName("yyy");

// 连接注册中心配置
RegistryConfig registry = new RegistryConfig();
registry.setAddress("zookeeper://172.16.16.13:2181");
//registry.setUsername("aaa");
//registry.setPassword("bbb");

ReferenceConfig<DemoService> reference = new ReferenceConfig<DemoService>(); // 此实例很重，封装了与注册中心的连接以及与提供者的连接，请自行缓存，否则可能造成内存和连接泄漏
reference.setApplication(application);
reference.setRegistry(registry); // 多个注册中心可以用setRegistries()
reference.setInterface(DemoService.class);
reference.setVersion("1.0.0");

// 和本地bean一样使用xxxService
DemoService xxxService = reference.get(); // 注意：此代理对象内部封装了所有通讯细节，对象较重，请缓存复用
String result = xxxService.sayHello("world");
System.out.println(result);
*/

    }
}
