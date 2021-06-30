### 代码结构
root
├── example                             # 存放样例代码  
│   ├── run_fx_quantize.py              # fx量化方式  
│   ├── run_quantize.py                 # 通用量化  
│   ├── train.py                        # 在cifar上训练一个简单模型  
│   ├── utils.py                        #   
│   └── zfnet.py                        # 网络结构  
├── qqquantize  
│   ├── observers                       # 各类observer以及量化Operations  
│   │   ├── fake_quantize.py  
│   │   ├── histogramobserver.py  
│   │   ├── minmaxobserver.py  
│   │   └── observerbase.py  
│   ├── qmodules                        # 量化版本的Operations  
│   │   ├── qbatchnorm.py  
│   │   ├── qconv.py  
│   │   ├── qlinear.py  
│   │   ├── qrelu.py  
│   │   └── qstub.py  
│   ├── qconfig.py                      # 一些默认的配置  
│   ├── qtensor.py  
│   ├── quantize.py                     # 浮点模型转换量化模型的一些函数  
│   ├── savehook.py                     # 用于保存中间数据的方法  
│   ├── toolbox  
│   │   └── fxquantize.py               # fx量化规则  
│   └── utils.py  
└── README.md  

### 代码逻辑
以run_fx_quantize.py为例，（此demo有些过时了，直接看yolo的train_quant吧）
1. 通过fuse_zfnet,融合bn层和conv层
2. 使用prepare函数，将网络的各层替换成量化版本的层
3. 调用enable_fake_quant和disable_observer函数。这时，网络不进行量化，只统计数据分布
4. 调用enable_fake_quant和disable_observer函数。网络开始进行量化，不再统计数据分布
5. 使用Qtensor进行一次inference获取各层input,weight,output的量化bit数，fx_adjust_bits函数根据规则调整bit，使其符合规则
6. 开始训练


### HistObserver
先使用minmaxObserver来统计tensor的最大值最小值，然后用swap_minmax_to_hist把minmaxObserver更换成histObserver
再在数据集上inference一遍，统计直方图数据。最后用calc_qparams计算scale
然而效果并不好

### MinmaxChannelsObserver
perchannel版本的minmaxOb