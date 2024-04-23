# PointLoc (pytorch)
the pytorch implementation of PointLoc based on [flownet3d_pytorch](https://github.com/hyangwinter/flownet3d_pytorch/tree/master)


## Installation

```shell
pip install -r requirements.txt
```

### pointnet2_pytoch install
[README.md](models%2Flib%2FREADME.md)

The implementation from yanx27/Pointnet_Pointnet2_pytorch
```shell
cd lib
python setup.py install
cd ../
```

#### installation Issue
Replacing THC/THC.h module to ATen/ATen.h module
https://stackoverflow.com/questions/72988735/replacing-thc-thc-h-module-to-aten-aten-h-module

## dataset
vReLoc dataset : https://github.com/loveoxford/vReLoc
```shell
$pointloc
├── dataset
│   └── vReLoc
│       └── full
│           ├── TestSplit.txt
│           ├── TrainSplit.txt
│           ├── seq-01
│           │   ├── frame-000000.bin
│           │   ├── frame-000000.color.png
│           │   ├── frame-000000.pose.txt
│           │   ├── ...
│           ├── seq-02
│           │   ├── frame-000000.bin
│           │   ├── frame-000000.color.png
│           │   ├── frame-000000.pose.txt
│           ├── ...
```



## Training

```shell
python main.py 
```

## Demo

<img src="etc%2Fpointloc_demo_onlyscan.gif" width="500" height="500"/>
