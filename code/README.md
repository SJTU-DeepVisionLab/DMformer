## DMformer: Difficulty-adapted Masked Transformer for Semi-Supervised Medical Image Segmentation

## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python == 3.6 
* Efficientnet-Pytorch `pip install efficientnet_pytorch`
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage

1. Clone the repo:
```
git clone https://github.com/SJTU-DeepVisionLab/DMformer.git
```
2. Download the processed data and put the data in `../data/AbdomenMR` or `../data/ACDC`, please read and follow the [README](https://github.com/Luoxd1996/SSL4MIS/tree/master/data/).

3. Train the model
```
cd code
python train_2D_fully_XXXXX.py
```

4. Test the model
```
python test_2D_fully_XXXXX.py
```
## Acknowledgement
* Part of the code is adapted from open-source codebase and original implementations of algorithms, we thank these author for their fantastic and efficient codebase, such as, [UA-MT](https://github.com/yulequan/UA-MT), [SwinUNet](https://github.com/HuCaoFighting/Swin-Unet) and [segmentatic_segmentation.pytorch](https://github.com/qubvel/segmentation_models.pytorch). 
