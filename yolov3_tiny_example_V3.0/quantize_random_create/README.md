# 随机通道代码
## 生成一个随机通道数的yolo模型，并导出onnx给软件部做对比测试用

### 文档结构
root
├── yolotiny_8bit_onnx               # 存放生成的随机通道样例onnx
├── export_onnx.py                   # 导出onnx代码  
├── yolotiny_random.py               # 随机通道yolov3tiny网络模型
├── yolotiny_rdmcreate.py            # 生成yolov3tiny随机通道代码   
└── README.md  

### 生成随机通道样例代码逻辑

#### 1、yolotiny_rdmcreate.py生成随机通道

生成随机通道函数为代码100~141行Random_chnl_crt函数。

代码128~139行：以原通道数为上限生成随机输出通道并以列表形式存储

#### 2、yolotiny_random.py生成随机通道网络模型

修改了原复现Yolov3tiny网络模型代码每层输入输出通道部分，代码26~47行将随机通道列表传入网络结构中。

#### 3、export_onnx.py导出onnx

代码60~64行：生成随机输入、随机通道列表以及随机数分类，并将这些参数传入随机通道网络模型
代码66~91行：将随机通道模型转换为量化模型
代码92~123行：将量化参数导出并存入未加量化结构的新网络结构
代码126~129行：导出onnx