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
├── demo.py                             # 将量化开关加入模型的简单样例代码    
└── README.md  

### 代码逻辑
以demo.py为例（此样例使用的为一个简单的示例网络）
1. 使用qconverter，将网络的各层替换成量化版本的层
2. 调用enable_fake_quant和disable_observer函数。这时，网络不进行量化，只统计数据分布
3. 调用enable_fake_quant和disable_observer函数。网络开始进行量化，不再统计数据分布
4. 训练之前需要调用enable_observer和enable_calc_qparams开关
5. 开始训练

### MinmaxChannelsObserver
perchannel版本的minmaxObserver

### 量化要求
目前量化训练要求为weight采用perchannel统计数据，act采用pertensor统计数据进行训练。
目前的试验结论得出，为了提高量化精度，需要在网络的最后一层weight和act均采用perchannel进行训练。此开关可以在qqquantize.quantize.py的_propagate_qconfig函数中添加，请参照该函数中的注释操作。