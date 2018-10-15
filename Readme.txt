程序分为node端和edge-server端

node端文件位于815sendNOde文件夹

使用了yolov2的模型
weights_loader 用来读取并加载加载yolov2的参数文件
net(net_max3)  中定义了tiny-yolov2的模型网络
new_test_node(server) 为主程序，先运行server端后运行node端传输文件（记得要改ip地址和端口号）

主程序分为cam和split版，cam为指定分割点使用cam来演示，split为使用一张图片处理，遍历所有分割点（示例图片为zzh.jpg）