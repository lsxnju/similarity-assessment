# i3d
#### model
- 只实验了RGB部分，train_rgb.py or rgb.ipynb
- main.py 中有flow部分，没试过

#### dataset
- hmdb51
- test.ipynb中可以进行检查数据集是否读对

#### 参数   
- model_weight 为I3D预训练参数，需要load除去最后一行的参数
- net中有更新参数，vf为最终版

#### loss-acc
- loss在计算的时候注意不要有出现取消梯度的步骤
- 计算得到的loss，存入list不能直接画图
    - train: "tensor(2.3797,grad_fn=<DivBackward0>)"
    - test: tensor(2.2151)
- img 有画到的图
