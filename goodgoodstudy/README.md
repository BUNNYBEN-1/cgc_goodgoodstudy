## README

[TOC]

#### 一、GitHub项目文件结构

```
cgc_goodgoodstudy
├── goodgoodstudy
│   	├── source_code.cpp
│   	├── makefile
│   	└── README
├── goodgoodstudy_report
│   	└── report.pdf
└── goodgoodstudy.exe
```



#### 二、测试方式

打开命令行，进入目录cgc_goodgoodstudy/goodgoodstudy。

```
cd cgc_goodgoodstudy/goodgoodstudy
```

编译makefile文件，在上级目录形成可执行文件。

```
make
```

返回上级目录。

```
cd ..
```

执行可执行文件，传入7个参数，分别是：输入顶点特征长度、第一层顶点特征长度、第二层顶点特征长度、图结构文件名、输入顶点特征矩阵文件名、第一层权重矩阵文件名、第二层权重矩阵文件名。

```
./goodgoodstudy.exe 64 16 8 graph/500K500K.txt embedding/500K500K.bin weight/W_64_16.bin weight/W_16_8.bin
```

可执行程序打印输出两个值，分别为最大的顶点特征矩阵行和执行时间。

<img src="C:\Users\bunny\AppData\Roaming\Typora\typora-user-images\image-20230720152539507.png" alt="image-20230720152539507" style="zoom:80%;" />
