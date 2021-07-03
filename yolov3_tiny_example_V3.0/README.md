### 文档结构
.
├── README.md                                       # 本说明文档                             
├── compare                                         # original U-Yolov3                             
├── data                                            # original U-Yolov3                             
├── detect.py                                       # original U-Yolov3                             
├── models                                          # original U-Yolov3                             
├── qqquantize                                      # 量化代码, 详情见内部README.md                             
├── quantize_export_onnx.py                         # 导出量化模型的onnx                             
├── quantize_export_onnx_txt_compare_ckpt_PCPT.py   # 导出量化模型的pt/onnx/txt/npy去比对, txt格式是V2, 多了load checkpoint的步骤                              
├── quantize_export_onnx_txt_compare_v3tiny_V1.py   # 导出量化模型的pt/onnx/txt/npy去比对, txt格式是V1：如: conv:[1.0, 2.0]                             
├── quantize_export_onnx_txt_compare_v3tiny_V2.py   # 导出量化模型的pt/onnx/txt/npy去比对, txt格式是V2：如: conv_pc:[1.0, 2.0]                             
├── quantize_random_create                          # 生成随机通道模型的工程, 详情见内部README.md                             
├── quantize_test.py                                # 量化模型测试map的代码, 应该没什么用，仅做保留                             
├── quantize_train.py                               # 量化训练的代码                             
├── quantize_utils.py                               # qconfig、导出、加载等一些函数工具                             
├── requirements.txt                                # original U-Yolov3                             
├── runs                                            # original U-Yolov3, 训练结果保存的目录                             
├── start_quantize_train.sh                         # 启动量化训练脚本                             
├── test.py                                         # original U-Yolov3, 和官方不同的地方是 img/255.0 改了                             
├── train.py                                        # original U-Yolov3, 和官方不同的地方是 img/255.0 改了                             
├── utils                                           # original U-Yolov3                             
└── weights                                         # original U-Yolov3                             
                             

### 使用

#### 1、量化训练

* 运行
    ```
        python quantize_train.py \
        --weights ./weights/voc_tiny_fp32_0.554.pt \
        --cfg ./models/yolov3-tiny.yaml \
        --data ./data/voc.yaml \
        --batch-size 8 \
        --device 0 \
        --name q_fix5_bn_concat_pchannel

        或者

        bash start_quantize_train.sh
    ```

#### 2、数据比对

* 运行
```
        python quantize_export_onnx_txt_compare_v3tiny_V1.py
    会在compare目录下生成一个初始随机模型的 txt & onnx & pt & npy

    如果要导出已经训好的模型，需要运行：
        python quantize_export_onnx_txt_compare_ckpt_PCPT.py
    注意修改这个代码里面的checkpoint路径
```


#### 3、导出onnx

* 运行
```
    python quantize_export_onnx.py
```

#### 4、随机通道
* 进入 quantize_random_create 按照 readme 操作
