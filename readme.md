# 适合保存大量小文件数据集的文件系统性能测试

## 测试方法

* 挂载一个tmpfs文件系统，在其中创建大量小文件。
* 在tmpfs中创建不同文件系统的镜像，将小文件复制入镜像中。
* 将镜像挂载
* 测试随机读取镜像中的小文件的吞吐量

## 测试对象

* 符号链接至tmpfs
* ext4
* squashfs (禁用所有压缩)
* squashfs (禁用数据压缩)
* erofs
* fuse-zip (zip仅存储)
* archivemount (zip仅存储)

## 环境准备

* Ubuntu 20.04 或更高
* 默认配置下，至少8G内存
* sudo权限

```bash
sudo apt install python3 python3-pip squashfs-tools erofs-utils zip fuse-zip archivemount
pip install -r requirments.txt
```
