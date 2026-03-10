# Sketch2Vox 





#### Clone the Code Repository

```
git clone https://github.com/hzxie/Pix2Vox.git
```

#### Install Python Denpendencies

```
cd sketch2Vox
pip install -r requirements.txt
```

#### Update Settings in `config.py`

You need to update the file path of the datasets:

```
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/path/to/Datasets/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH            = '/path/to/Datasets/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'
__C.DATASETS.PASCAL3D.ANNOTATION_PATH       = '/path/to/Datasets/PASCAL3D/Annotations/%s_imagenet/%s.mat'
__C.DATASETS.PASCAL3D.RENDERING_PATH        = '/path/to/Datasets/PASCAL3D/Images/%s_imagenet/%s.JPEG'
__C.DATASETS.PASCAL3D.VOXEL_PATH            = '/path/to/Datasets/PASCAL3D/CAD/%s/%02d.binvox'
__C.DATASETS.PIX3D.ANNOTATION_PATH          = '/path/to/Datasets/Pix3D/pix3d.json'
__C.DATASETS.PIX3D.RENDERING_PATH           = '/path/to/Datasets/Pix3D/img/%s/%s.%s'
__C.DATASETS.PIX3D.VOXEL_PATH               = '/path/to/Datasets/Pix3D/model/%s/%s/%s.binvox'
```

## Get Started

To train, you can simply use the following command:

```
python3 runner.py
```

To test, you can use the following command:

```
python3 runner.py --test --weights=/path/to/pretrained/model.pth
```



## License

This project is open sourced under MIT license.
