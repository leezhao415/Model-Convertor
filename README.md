#### 1 Pytorch 2 ONNX 2 ONNX-sim

```python
pytorch2onnx2onnx_sim2onnx_infer.py
```

`执行前参数修改`：

```bash
## -------------------------------------------------------------------------------------------------------------------------------- ##
# ONNX模型输出路径
parser.add_argument("--output-name", type=str, default="./Outputs/model_onnx/yolox_nano_112_300_20220126.onnx", help="output name of models")
# ONNX简化模型输出路径
parser.add_argument("--output-name-sim", type=str, default="./Outputs/model_onnx_sim/yolox_nano_112_300_20220126_sim.onnx", help="output name of models-sim")
# config_file
parser.add_argument("-f","--exp_file",default='./ModelFile/yolox/yolox_voc_nano.py',type=str,help="expriment description file",)
# model_file
parser.add_argument("-c", "--ckpt", default='./ModelFile/yolox_nano_112_300_20220126.pth', type=str, help="ckpt path")
## -------------------------------------------------------------------------------------------------------------------------------- ##


## ----------------ONNX model_infer------------------------------------------------------------------------------------------ ##
# Path to your input image.
parser.add_argument("-i","--image_path",type=str,default='./TestFiles/obj365_train_000000026463.jpg',help="Path to your input image.",)
# Path to your output directory.
parser.add_argument("--output_dir",type=str,default='./Outputs/demo_output_onnx_infer',help="Path to your output directory.",)
# Score threshould
parser.add_argument("-s","--score_thr",type=float,default=0.3,help="Score threshould to filter the result.",)
# input shape
parser.add_argument("--input_shape",type=str,default="416,416",help="Specify an input shape for inference.",)
parser.add_argument("--with_p6",action="store_true",help="Whether your model uses p6 in FPN/PAN.",)
    ## -------------------------------------------------------------------------------------------------------------------------------- ##
```

#### 2 NCNN模型-onnx2ncnn

```python
./ncnn/build/tools/onnx/onnx2ncnn your_path/yolox_nano_sim.onnx your_path/yolox_nano.param your_path/yolox_nano.bin
```

#### 3 NCNN模型优化-ncnnoptimize

```python
./ncnn/build/tools/ncnnoptimize model.param model.bin model-opt.param model-opt.bin 0
```

#### 4 模型秒转化代码（pth2onnx2onnx_sim2ncnn2ncnn _optimize）

```python
# 进入该工程目录
source activate open-mmlab   # 进入虚拟环境（包含pytorch、onnx、onnxsim等模块）

## 重要：将待转化pytorch模型放入./ModelFile/文件夹下，并修改模型名称为model.pth

# pth转化onnx及onnx-smi
python pytorch2onnx2onnx_sim2onnx_infer.py  

# onnx2ncnn
./ncnn/build/tools/onnx/onnx2ncnn ./Outputs/model_onnx_sim/model_sim.onnx ./Outputs/model_ncnn/model_sim.param ./Outputs/model_ncnn/model_sim.bin

# 将./Outputs/model_ncnn/model_sim.param中的Split-Crop-Concat修改为YoloV5Focus，并将第二行中的总层数（第一个参数）对应修改。

# ncnn2ncnn optimize
./ncnn/build/tools/ncnnoptimize ./Outputs/model_ncnn/model_sim.param ./Outputs/model_ncnn/model_sim.bin ./Outputs/model_ncnn/model-opt.param ./Outputs/model_ncnn/model-opt.bin 0
```

